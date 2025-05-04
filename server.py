#!/usr/bin/env python3
"""
Unified MLX Vision-Language server
• /v1/chat/completions  (OpenAI compatible)
• /v1/responses         (Azure/OpenAI “responses” API)
Supports images (url | base64 | path) and video key-frames.
"""

import os, io, re, json, time, uuid, math, base64, tempfile, subprocess, asyncio, logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

import uvicorn, requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import mlx.core as mx                   # CPU/GPU device picked automatically (Apple Silicon ok)
from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import stream_generate

# ───────────────────────────── CLI ───────────────────────────────
import argparse
parser = argparse.ArgumentParser(description="MLX VL-server")
parser.add_argument("-m","--model", default="mlx-community/Qwen2-VL-2B-Instruct-4bit")
parser.add_argument("--port", type=int, default=8005)
parser.add_argument("--strip-thinking", action="store_true")
parser.add_argument("--ffmpeg", default="ffmpeg")
parser.add_argument("--default-frames", type=int, default=4)
parser.add_argument("--patch-cap", action="store_true",
                    help="Rescale images so #32-px patches ≤ 1536 (OpenAI style)")
args = parser.parse_args()

MAX_NEW_TOKENS = 512
PATCH_LIMIT = 1536
PATCH_SIZE  = 32
STRIP_RE    = re.compile(r"<think>.*?</think>", re.I|re.S)
LOG         = logging.getLogger("vlm-srv")
logging.basicConfig(level=logging.INFO)

# ────────────────────── model / processor load ───────────────────
def load_model_and_processor(model_path:str):
    lower = model_path.lower()
    if "qwen" in lower and "vl" in lower:
        # use qwen-vl-utils AutoProcessor (handles dynamic resize, bbox return, etc.)
        from transformers import AutoProcessor, AutoModelForVision2Seq
        LOG.info("Detected Qwen-VL – loading with transformers AutoProcessor")
        model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor
    # default MLX loader (covers Llava-1.6, BLIP-2, etc.)
    cfg_path = os.path.join(model_path,"config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    return load(model_path, **cfg)

MODEL, PROCESSOR = load_model_and_processor(args.model)
MODEL_NAME = os.path.basename(args.model) if os.path.exists(args.model) else args.model
LOG.info("Loaded model %s", MODEL_NAME)

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

class RespReq(BaseModel):
    model:str = MODEL_NAME
    input:Union[str,List[Dict[str,Any]]]
    previous_response_id:Optional[str] = None
    stream:bool = False
    temperature:float = 1.0
    top_p:float = 1.0
    max_output_tokens:Optional[int] = None
    images:List[str] = Field(default_factory=list)
    videos:List[Dict[str,Any]] = Field(default_factory=list)
    strip_thinking:Optional[bool] = None
    stripThinking:Optional[bool] = None

class Usage(BaseModel):
    input_tokens:int
    output_tokens:int
    total_tokens:int
    tokens_per_second:float

# ─────────────────────── helpers ─────────────────────────────────
STATE:Dict[str,List[Dict[str,str]]] = {}

def _strip_thinks(text:str, on:bool)->str:
    return STRIP_RE.sub("", text) if on else text

def _apply_patch_cap(img:Image.Image)->Image.Image:
    if not args.patch_cap: return img
    w,h = img.size
    patches = math.ceil(w/PATCH_SIZE)*math.ceil(h/PATCH_SIZE)
    if patches<=PATCH_LIMIT: return img
    scale = math.sqrt(PATCH_LIMIT/patches)
    new_w,new_h = int(w*scale), int(h*scale)
    return img.resize((new_w,new_h), Image.BICUBIC)

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
        # evenly spaced frames
        out_dir = tempfile.mkdtemp()
        cmd = [args.ffmpeg,"-y","-i",tmp.name,
               "-vf",f"fps={n}/$(ffprobe -v 0 -show_entries format=duration "
                     f"-of default=nw=1:nk=1 {tmp.name})",
               os.path.join(out_dir,"%04d.png")]
        subprocess.run(" ".join(cmd), shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        files = sorted(os.listdir(out_dir))[:n]
        return [_apply_patch_cap(Image.open(os.path.join(out_dir,f)).convert("RGB"))
                for f in files]

def collect_images(images:List[str], videos:List[Dict[str,Any]])->List[Image.Image]:
    out=[]
    for ref in images:
        out.append(_load_one_image(ref))
    for vid in videos:
        frames = vid.get("frames", args.default_frames)
        out.extend(_video_to_frames(vid["url"], frames))
    return out

def build_prompt(msgs:List[Dict], num_imgs:int)->str:
    return apply_chat_template(
        PROCESSOR,
        config=getattr(MODEL,"config",{}).__dict__ if hasattr(MODEL,"config") else {},
        prompt=msgs,
        num_images=num_imgs
    )

def sync_generate(msgs:List[Dict], pil_imgs:List[Image.Image],
                  max_toks:int,temp:float,top_p:float,strip:bool):
    prompt = build_prompt(msgs, len(pil_imgs))
    t0=time.time()
    out = generate(MODEL, PROCESSOR, prompt=prompt, image=pil_imgs,
                   max_tokens=max_toks, temp=temp, top_p=top_p, verbose=False)
    out = _strip_thinks(out, strip)
    dt=time.time()-t0; usage=Usage(input_tokens=len(prompt),
                                   output_tokens=len(out),
                                   total_tokens=len(prompt)+len(out),
                                   tokens_per_second=(len(prompt)+len(out))/max(dt,1e-6))
    return out,usage

async def stream_generate_chunks(req_id:str, prompt:str, pil_imgs:List[Image.Image],
                                 max_toks:int,temp:float,top_p:float,
                                 strip:bool, created:int)->AsyncGenerator[str,None]:
    inside=False; seen=0
    async for res in stream_generate(MODEL,PROCESSOR,
                                     prompt,image=pil_imgs,
                                     max_tokens=max_toks,temp=temp,top_p=top_p):
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
        payload={"id":req_id,"object":"chat.completion.chunk","created":created,
                 "model":MODEL_NAME,"choices":[{"index":0,"delta":{"content":chunk},
                                                "finish_reason":None}]}
        yield f"data: {json.dumps(payload,ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)

# ─────────────────────────── app ─────────────────────────────────
app = FastAPI()

def _strip_flag(body)->bool:
    return bool(getattr(body,"strip_thinking",False) or
                getattr(body,"stripThinking",False) or
                args.strip_thinking)

@app.post("/v1/chat/completions")
async def chat_completions(body:ChatReq):
    if body.model!=MODEL_NAME:
        raise HTTPException(400,f"Model {body.model} not loaded.")
    pil_imgs = collect_images(body.images, body.videos)
    strip=_strip_flag(body)
    if not body.stream:
        txt,usage = sync_generate([m.model_dump() for m in body.messages], pil_imgs,
                                  body.max_tokens, body.temperature, body.top_p, strip)
        return {"id":f"chatcmpl-{uuid.uuid4()}","object":"chat.completion",
                "created":int(time.time()),"model":MODEL_NAME,
                "choices":[{"index":0,"message":{"role":"assistant","content":txt},
                            "finish_reason":"stop"}],
                "usage":usage.model_dump()}
    prompt = build_prompt([m.model_dump() for m in body.messages], len(pil_imgs))
    req_id=f"chatcmpl-{uuid.uuid4()}"; created=int(time.time())
    gen=stream_generate_chunks(req_id,prompt,pil_imgs,
                               body.max_tokens,body.temperature,body.top_p,
                               strip,created)
    return StreamingResponse(gen,media_type="text/event-stream")

@app.post("/v1/responses")
async def responses(body:RespReq):
    if body.model!=MODEL_NAME:
        raise HTTPException(400,"unknown model")
    history=STATE.get(body.previous_response_id,[])
    if isinstance(body.input,str):
        history.append({"role":"user","content":body.input})
    else:
        history += [{"role":m.get("role","user"),"content":m.get("content","")}
                    for m in body.input]
    pil_imgs = collect_images(body.images, body.videos)
    strip=_strip_flag(body)
    max_tok = body.max_output_tokens or MAX_NEW_TOKENS
    if not body.stream:
        txt,usage = sync_generate(history,pil_imgs,max_tok,
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
    prompt = build_prompt(history, len(pil_imgs))
    resp_id=f"resp_{uuid.uuid4().hex}"; created=int(time.time())
    async def gen():
        async for chunk in stream_generate_chunks(resp_id,prompt,pil_imgs,
                                                  max_tok,body.temperature,body.top_p,
                                                  strip,created):
            yield chunk
        STATE[resp_id]=history
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(),media_type="text/event-stream")

# ───────────────────────── main ──────────────────────────────────
if __name__ == "__main__":
    LOG.info("Server ready on :%d  (model %s)", args.port, MODEL_NAME)
    uvicorn.run(app, host="0.0.0.0", port=args.port)