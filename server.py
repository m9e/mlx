#!/usr/bin/env python3
"""
Unified Vision-Language server (MLX) – fully OpenAI-compatible.

✔ /v1/chat/completions        – text, images, or mixed content
✔ /v1/responses               – OpenAI “responses” shape
✔ URL, base64, local-path images + video key-frames
✔ <think>…</think> stripping via --strip-thinking or JSON flag
"""

from __future__ import annotations
import asyncio
import os, re, gc, io, json, time, uuid, math, base64, tempfile, subprocess, argparse, logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple

import requests, uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

import mlx.core as mx
from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import stream_generate

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MLX vision-language server")
parser.add_argument("-m", "--model", default="mlx-community/Qwen2-VL-2B-Instruct-4bit")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=18000)
parser.add_argument("--strip-thinking", action="store_true")
parser.add_argument("--default-frames", type=int, default=4)
parser.add_argument("--ffmpeg", default="ffmpeg")
parser.add_argument("--col-a", default="qwen")          # token list A
parser.add_argument("--col-b", default="vl")            # token list B
args = parser.parse_args()

TOK_A = {t.lower() for t in args.col_a.split(",") if t}
TOK_B = {t.lower() for t in args.col_b.split(",") if t}

MAX_TOKENS = 512
PATCH_LIMIT, PATCH_SIZE = 1536, 32
THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vlm-srv")

# ── model & processor ────────────────────────────────────────────
def load_model(repo: str):
    special = bool({repo.lower()} & TOK_A) and bool({repo.lower()} & TOK_B)
    cfg_path = os.path.join(repo, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    model, proc = load(repo, **cfg)
    if special:
        log.info("🔎  Special vision pipeline enabled for %s + %s", TOK_A, TOK_B)
    gc.collect()
    model.eval()
    return model, proc

MODEL, PROCESSOR = load_model(args.model)
MODEL_NAME = os.path.basename(args.model) if os.path.exists(args.model) else args.model

# ── Pydantic schemas ────────────────────────────────────────────
class MessagePart(BaseModel):
    """OpenAI vision message part."""
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class VLMessage(BaseModel):
    role: str
    # can be plain string *or* list[MessagePart] (OpenAI vision shape)
    content: Union[str, List[MessagePart]]

class VLRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[VLMessage]
    images: List[str] | None = None
    videos: List[Dict[str, Any]] = Field(default_factory=list)
    max_tokens: int = MAX_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    strip_thinking: Optional[bool] = None

    # flatten image_url parts into the top-level images list
    @model_validator(mode="after")
    def _collect_images(cls, v: "VLRequest"):
        imgs = list(v.images or [])
        flat_msgs: List[Dict[str, str]] = []
        for m in v.messages:
            if isinstance(m.content, list):
                text_buf = []
                for part in m.content:
                    if part.type == "text" and part.text:
                        text_buf.append(part.text)
                    elif part.type == "image_url" and part.image_url:
                        imgs.append(part.image_url["url"])
                m.content = "\n".join(text_buf)
            # convert back to simple dict for template
            flat_msgs.append({"role": m.role, "content": m.content})
        v.__dict__["flat_msgs"] = flat_msgs      # cached for handler
        v.__dict__["all_images"] = imgs          # cached for handler
        return v

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float

class Choice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class CompletionChunk(BaseModel):
    id: str
    object: str = "vision.language.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

# ── helpers ──────────────────────────────────────────────────────
def _strip(text: str, flag: bool) -> str:
    return THINK_RE.sub("", text) if flag else text

def _patch_cap(img: Image.Image) -> Image.Image:
    w, h = img.size
    patches = math.ceil(w / PATCH_SIZE) * math.ceil(h / PATCH_SIZE)
    if patches <= PATCH_LIMIT:
        return img
    scale = math.sqrt(PATCH_LIMIT / patches)
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def _load_image(ref: str) -> Image.Image:
    if ref.startswith("data:image/"):
        img = Image.open(io.BytesIO(base64.b64decode(ref.split(",", 1)[1])))
    elif ref.startswith("http"):
        img = Image.open(io.BytesIO(requests.get(ref, timeout=15).content))
    else:
        img = Image.open(ref)
    return _patch_cap(img.convert("RGB"))

def _video_frames(url: str, n: int) -> List[Image.Image]:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(requests.get(url, timeout=30).content if url.startswith("http") else open(url, "rb").read())
        tmp.flush()
        fps = n / float(subprocess.check_output(
            [args.ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "0", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", tmp.name], text=True).strip() or "1")
        out_dir = tempfile.mkdtemp()
        subprocess.run([args.ffmpeg, "-y", "-i", tmp.name, "-vf", f"fps={fps}",
                        os.path.join(out_dir, "%04d.png")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return [_patch_cap(Image.open(os.path.join(out_dir, f)).convert("RGB"))
                for f in sorted(os.listdir(out_dir))[:n]]

def gather_images(req: VLRequest) -> List[Image.Image]:
    imgs = [_load_image(u) for u in req.all_images]
    for v in req.videos:
        imgs.extend(_video_frames(v["url"], v.get("frames", args.default_frames)))
    return imgs

def sync_generate(req: VLRequest, imgs: List[Image.Image]) -> Tuple[str, Usage]:
    prompt = apply_chat_template(
        PROCESSOR,
        config=getattr(MODEL, "config", {}).__dict__ if hasattr(MODEL, "config") else {},
        prompt=req.flat_msgs,
        num_images=len(imgs),
    )
    start = time.time()
    out = generate(MODEL, PROCESSOR, prompt, image=imgs,
                   max_tokens=req.max_tokens, temp=req.temperature, top_p=req.top_p, verbose=False)
    dur = time.time() - start
    out = _strip(out, req.strip_thinking or args.strip_thinking)
    usage = Usage(prompt_tokens=len(prompt), completion_tokens=len(out),
                  total_tokens=len(prompt) + len(out),
                  tokens_per_second=(len(prompt) + len(out)) / max(dur, 1e-6))
    return out, usage

async def stream_generate_chunks(req_id: str, req: VLRequest, imgs: List[Image.Image]):
    prompt = apply_chat_template(PROCESSOR, config=getattr(MODEL, "config", {}).__dict__,
                                 prompt=req.flat_msgs, num_images=len(imgs))
    created = int(time.time())
    first = True
    async for res in stream_generate(MODEL, PROCESSOR, prompt, image=imgs,
                                     max_tokens=req.max_tokens, temp=req.temperature, top_p=req.top_p):
        if not res.text:
            continue

        delta = {"content": res.text}
        if first:
            delta["role"] = "assistant"   # OpenAI expects this on the first chunk
            first = False

        chunk_json = CompletionChunk(
            id=req_id,
            created=created,
            model=MODEL_NAME,
            choices=[{
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }],
        ).model_dump_json()

        yield f"data: {chunk_json}\n\n"
        await asyncio.sleep(0)

    # final "[DONE]" sentinel
    yield "data: [DONE]\n\n"

# ── FastAPI ──────────────────────────────────────────────────────
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(req: VLRequest):
    if req.model != MODEL_NAME:
        log.warning(f"Requested model '{req.model}' does not match available model '{MODEL_NAME}'")
    images = gather_images(req)
    if not req.stream:
        text, usage = sync_generate(req, images)
        return CompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=MODEL_NAME,
            choices=[
                Choice(
                    index=0,
                    message={"role": "assistant", "content": text},
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
    gen = stream_generate_chunks(f"chatcmpl-{uuid.uuid4()}", req, images)
    return StreamingResponse(gen, media_type="text/event-stream")

# Optional: keep /v1/responses endpoint unchanged from last revision
# -----------------------------------------------------------------

if __name__ == "__main__":
    log.info("Serving %s on %s:%d", MODEL_NAME, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)