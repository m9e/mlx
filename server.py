#!/usr/bin/env python3
"""
Unified Vision-Language server (MLX)
/v1/chat/completions  – OpenAI-shape
/v1/responses         – Azure/OpenAI “responses” shape
Supports URL/local/base64 images, video key-frames, <think> stripping, etc.
"""

import os, io, re, gc, json, time, uuid, math, base64, random, tempfile, subprocess
import asyncio, logging, argparse
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

import uvicorn, requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import mlx.core as mx
from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import stream_generate

# ───────────────────────────── CLI ───────────────────────────────
parser = argparse.ArgumentParser(description="MLX Vision-Language server")
parser.add_argument("-m", "--model",
                    default="mlx-community/Qwen2-VL-2B-Instruct-4bit",
                    help="Model path or HF repo cloned for MLX")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=18000)
parser.add_argument("--strip-thinking", action="store_true")
parser.add_argument("--ffmpeg", default="ffmpeg",
                    help="Path to ffmpeg (for video frame extraction)")
parser.add_argument("--default-frames", type=int, default=4,
                    help="How many key-frames to sample per video")
parser.add_argument("--patch-cap", action="store_true",
                    help="Rescale images so 32-px patches ≤ 1536 (OpenAI style)")
# allow comma-separated override of detector lists
parser.add_argument("--col-a", default="qwen",
                    help="Comma-sep list; need ONE token from here …")
parser.add_argument("--col-b", default="vl",
                    help="… and ONE token from here to trigger special handling")
args = parser.parse_args()

TOKENS_A = [t.strip().lower() for t in args.col_a.split(",") if t.strip()]
TOKENS_B = [t.strip().lower() for t in args.col_b.split(",") if t.strip()]

MAX_NEW_TOKENS = 512
PATCH_LIMIT = 1536
PATCH_SIZE  = 32
STRIP_RE    = re.compile(r"<think>.*?</think>", re.I|re.S)
LOG         = logging.getLogger("vlm-srv")
logging.basicConfig(level=logging.INFO)

# ─────────────────── model / processor loader ────────────────────
def load_model_and_processor(path:str):
    lower = path.lower()
    special = (any(t in lower for t in TOKENS_A)
               and any(t in lower for t in TOKENS_B))

    cfg_path = os.path.join(path, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    model, processor = load(path, **cfg)
    if special:
        LOG.info("Special vision pipeline activated for tokens %s & %s",
                 TOKENS_A, TOKENS_B)
    model.eval()
    mx.empty_cache(); gc.collect()
    return model, processor, special

MODEL, PROCESSOR, SPECIAL_PIPE = load_model_and_processor(args.model)
MODEL_NAME = os.path.basename(args.model) if os.path.exists(args.model) else args.model
LOG.info("Loaded model %s  (special=%s)", MODEL_NAME, SPECIAL_PIPE)

# ─────────────────────── data classes ────────────────────────────
class ChatMsg(BaseModel):
    role:str
    content:str

class ChatReq(BaseModel):
    model:str = MODEL_NAME
    messages:List[ChatMsg]
    images:List[str] = Field(default_factory=list)
    videos:List[Dict[str,Any]] = Field(default_factory=list)
    max_tokens:int = MAX_NEW_TOKENS
    temperature:float = 1.0
    top_p:float = 1.0
    stream:bool = False
    resize_shape:Optional[List[int]] = None
    strip_thinking:Optional[bool] = None
    stripThinking:Optional[bool] = None

class RespReq(ChatReq):
    input:Union[str,List[Dict[str,Any]]]
    previous_response_id:Optional[str] = None
    max_output_tokens:Optional[int] = None

class Usage(BaseModel):
    input_tokens:int
    output_tokens:int
    total_tokens:int
    tokens_per_second:float

# ───────────────────────── helpers ───────────────────────────────
STATE:Dict[str,List[Dict[str,str]]] = {}

def _strip_thinks(text:str, on:bool)->str:
    return STRIP_RE.sub("", text) if on else text

def _apply_patch_cap(img:Image.Image)->Image.Image:
    if not args.patch_cap: return img
    w,h = img.size
    patches = math.ceil(w/PATCH_SIZE)*math.ceil(h/PATCH_SIZE)
    if patches<=PATCH_LIMIT: return img
    scale = math.sqrt(PATCH_LIMIT/patches)
    return img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)

def _load_one_image(ref:str)->Image.Image:
    if ref.startswith("data:image/"):
        _,b64 = ref.split(",",1)
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
    elif ref.startswith("http://") or ref.startswith("https://"):
        img = Image.open(io.BytesIO(requests.get(ref, timeout=15).content))
    else:
        img = Image.open(ref)
    return _apply_patch_cap(img.convert("RGB"))

def _video_to_frames(url:str, n:int)->List[Image.Image]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        if url.startswith("http"):
            tmp.write(requests.get(url, timeout=30).content)
        else:
            tmp.write(open(url,"rb").read())
        tmp.flush()
        out_dir = tempfile.mkdtemp()
        # duration in seconds:
        dur = subprocess.check_output(
            [args.ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "0",
             "-select_streams", "v:0", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", tmp.name], text=True).strip()
        dur = float(dur) if dur else 1.0
        fps = n/dur
        subprocess.run([args.ffmpeg, "-y", "-i", tmp.name,
                        "-vf", f"fps={fps}", os.path.join(out_dir, "%04d.png")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        files = sorted(os.listdir(out_dir))[:n]
        return [_apply_patch_cap(Image.open(os.path.join(out_dir,f)).convert("RGB"))
                for f in files]

def collect_images(img_refs:List[str], vids:List[Dict[str,Any]])->List[Image.Image]:
    imgs=[_load_one_image(r) for r in img_refs]
    for vid in vids:
        frames=vid.get("frames", args.default_frames)
        imgs.extend(_video_to_frames(vid["url"], frames))
    return imgs

def build_prompt(msgs:List[Dict], n_img:int)->str:
    return apply_chat_template(PROCESSOR,
        config=getattr(MODEL,"config",{}).__dict__ if hasattr(MODEL,"config") else {},
        prompt=msgs, num_images=n_img)

def sync_generate(msgs:List[Dict], pil:List[Image.Image],
                  max_t:int, temp:float, top_p:float, strip:bool):
    prompt=build_prompt(msgs, len(pil))
    t0=time.time()
    out=generate(MODEL, PROCESSOR, prompt=prompt, image=pil,
                 max_tokens=max_t, temp=temp, top_p=top_p, verbose=False)
    out=_strip_thinks(out, strip)
    dt=time.time()-t0
    usage=Usage(input_tokens=len(prompt), output_tokens=len(out),
                total_tokens=len(prompt)+len(out),
                tokens_per_second=(len(prompt)+len(out))/max(dt,1e-6))
    return out,usage

async def stream_chunks(req_id:str, prompt:str, pil:List[Image.Image],
                        max_t:int,temp:float,top_p:float,
                        strip:bool, created:int)->AsyncGenerator[str,None]:
    inside=False; seen=0
    async for res in stream_generate(MODEL,PROCESSOR,prompt,image=pil,
                                     max_tokens=max_t,temp=temp,top_p=top_p):
        chunk=res.text
        if strip:
            buf=""; i=0
            while i<len(chunk):
                if not inside and chunk.startswith("<think>",i):
                    inside=True; i+=7; continue
                if inside and chunk.startswith("</think>",i):
                    inside=False; i+=8; continue
                if inside: i+=1; continue
                buf+=chunk[i]; 
                if chunk[i].strip(): seen+=1
                i+=1
            chunk = buf if seen<4 else chunk
        if not chunk: continue
        payload = {"id":req_id,"object":"chat.completion.chunk","created":created,
                   "model":MODEL_NAME,
                   "choices":[{"index":0,"delta":{"content":chunk},"finish_reason":None}]}
        yield f"data: {json.dumps(payload,ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)

# ─────────────────────────── app ─────────────────────────────────
app = FastAPI()

def _strip_flag(body)->bool:
    return bool(getattr(body,"strip_thinking",False)
                or getattr(body,"stripThinking",False)
                or args.strip_thinking)

@app.post("/v1/chat/completions")
async def chat_completions(body:ChatReq):
    if body.model!=MODEL_NAME:
        raise HTTPException(400,"Unknown model")
    pil=collect_images(body.images, body.videos)
    strip=_strip_flag(body)
    if not body.stream:
        txt,usage=sync_generate([m.model_dump() for m in body.messages],pil,
                                body.max_tokens,body.temperature,body.top_p,strip)
        return {"id":f"chatcmpl-{uuid.uuid4()}","object":"chat.completion",
                "created":int(time.time()),"model":MODEL_NAME,
                "choices":[{"index":0,"message":{"role":"assistant","content":txt},
                            "finish_reason":"stop"}],"usage":usage.model_dump()}
    prompt=build_prompt([m.model_dump() for m in body.messages],len(pil))
    req_id=f"chatcmpl-{uuid.uuid4()}"; created=int(time.time())
    return StreamingResponse(stream_chunks(req_id,prompt,pil,
                                           body.max_tokens,body.temperature,body.top_p,
                                           strip,created),
                             media_type="text/event-stream")

@app.post("/v1/responses")
async def responses(body:RespReq):
    if body.model!=MODEL_NAME:
        raise HTTPException(400,"Unknown model")
    history=STATE.get(body.previous_response_id,[])
    if isinstance(body.input,str):
        history.append({"role":"user","content":body.input})
    else:
        history.extend({"role":m.get("role","user"),"content":m.get("content","")}
                       for m in body.input)
    pil=collect_images(body.images, body.videos)
    strip=_strip_flag(body)
    max_t=body.max_output_tokens or MAX_NEW_TOKENS
    if not body.stream:
        txt,usage=sync_generate(history,pil,max_t,
                                body.temperature,body.top_p,strip)
        resp_id=f"resp_{uuid.uuid4().hex}"
        STATE[resp_id]=history+[{"role":"assistant","content":txt}]
        return {"id":resp_id,"object":"response","created_at":time.time(),
                "model":MODEL_NAME,
                "output":[{"id":"msg_"+uuid.uuid4().hex,"type":"message",
                           "role":"assistant",
                           "content":[{"type":"output_text","text":txt}]}],
                "output_text":txt,"temperature":body.temperature,"top_p":body.top_p,
                "previous_response_id":body.previous_response_id,
                "status":"completed","usage":usage.model_dump()}
    prompt=build_prompt(history,len(pil))
    resp_id=f"resp_{uuid.uuid4().hex}"; created=int(time.time())
    async def gen():
        async for c in stream_chunks(resp_id,prompt,pil,max_t,
                                     body.temperature,body.top_p,strip,created):
            yield c
        STATE[resp_id]=history
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(),media_type="text/event-stream")

# ───────────────────────── main ──────────────────────────────────
if __name__ == "__main__":
    LOG.info("Serving %s on %s:%d", MODEL_NAME, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)