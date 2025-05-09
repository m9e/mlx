#!/usr/bin/env python3
"""
Vision-Language inference server (MLX)
• /v1/chat/completions ― OpenAI-shape
• /v1/responses         ― Azure/OpenAI “responses” shape
Features: URL/base64/local images, video key-frames, <think> stripping, patch-cap, etc.
"""

import os, io, re, gc, json, time, uuid, math, base64, tempfile, subprocess, argparse, logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

import uvicorn, requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import mlx.core as mx
from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import stream_generate

# ── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MLX VL server")
parser.add_argument("-m","--model", default="mlx-community/Qwen2-VL-2B-Instruct-4bit")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=18000)
parser.add_argument("--strip-thinking", action="store_true")
parser.add_argument("--ffmpeg", default="ffmpeg")
parser.add_argument("--default-frames", type=int, default=4)
parser.add_argument("--patch-cap", action="store_true",
                    help="Scale so 32-px patches ≤ 1536 (OpenAI rule)")
parser.add_argument("--col-a", default="qwen")
parser.add_argument("--col-b", default="vl")
args = parser.parse_args()

TOKENS_A = [t.strip().lower() for t in args.col_a.split(",") if t.strip()]
TOKENS_B = [t.strip().lower() for t in args.col_b.split(",") if t.strip()]

MAX_NEW_TOKENS = 512
PATCH_LIMIT, PATCH_SIZE = 1536, 32
STRIP_RE = re.compile(r"<think>.*?</think>", re.I|re.S)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("vlm-srv")

# ── model & processor ────────────────────────────────────────────
def load_model_and_processor(path:str):
    lower = path.lower()
    special = (any(t in lower for t in TOKENS_A)
               and any(t in lower for t in TOKENS_B))
    cfg_path = os.path.join(path,"config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    model, processor = load(path, **cfg)
    if special:
        LOG.info("Special vision pipeline activated for tokens %s & %s", TOKENS_A, TOKENS_B)
    model.eval()
    gc.collect()
    return model, processor

MODEL, PROCESSOR = load_model_and_processor(args.model)
MODEL_NAME = os.path.basename(args.model) if os.path.exists(args.model) else args.model
LOG.info("Loaded model %s", MODEL_NAME)

# ── request / response schemas ───────────────────────────────────
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
    strip_thinking:Optional[bool] = None
    stripThinking:Optional[bool] = None

class RespReq(ChatReq):
    input:Union[str,List[Dict[str,Any]]]
    previous_response_id:Optional[str] = None
    max_output_tokens:Optional[int] = None

class Usage(BaseModel):
    input_tokens:int; output_tokens:int; total_tokens:int; tokens_per_second:float

# ── helpers ──────────────────────────────────────────────────────
STATE:Dict[str,List[Dict[str,str]]] = {}

def _strip(text:str, on:bool)->str: return STRIP_RE.sub("", text) if on else text

def _patch_cap(img:Image.Image)->Image.Image:
    if not args.patch_cap: return img
    w,h = img.size
    patches = math.ceil(w/PATCH_SIZE)*math.ceil(h/PATCH_SIZE)
    if patches<=PATCH_LIMIT: return img
    scale = math.sqrt(PATCH_LIMIT/patches)
    return img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)

def _load_img(ref:str)->Image.Image:
    if ref.startswith("data:image/"):
        img = Image.open(io.BytesIO(base64.b64decode(ref.split(",",1)[1])))
    elif ref.startswith("http"):
        img = Image.open(io.BytesIO(requests.get(ref, timeout=15).content))
    else:
        img = Image.open(ref)
    return _patch_cap(img.convert("RGB"))

def _video_frames(url:str, n:int)->List[Image.Image]:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(requests.get(url, timeout=30).content if url.startswith("http") else open(url,"rb").read())
        tmp.flush()
        dur=float(subprocess.check_output(
            [args.ffmpeg.replace("ffmpeg","ffprobe"),"-v","0","-select_streams","v:0",
             "-show_entries","format=duration","-of","default=nw=1:nk=1",tmp.name],text=True).strip() or "1")
        fps=n/dur
        out_dir=tempfile.mkdtemp()
        subprocess.run([args.ffmpeg,"-y","-i",tmp.name,"-vf",f"fps={fps}",
                        os.path.join(out_dir,"%04d.png")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return [_patch_cap(Image.open(os.path.join(out_dir,f)).convert("RGB"))
                for f in sorted(os.listdir(out_dir))[:n]]

def collect_images(imgs:List[str], vids:List[Dict[str,Any]])->List[Image.Image]:
    out=[_load_img(r) for r in imgs]
    for v in vids: out.extend(_video_frames(v["url"], v.get("frames", args.default_frames)))
    return out

def prompt_from_msgs(msgs:List[Dict], n_img:int)->str:
    return apply_chat_template(PROCESSOR,
        config=getattr(MODEL,"config",{}).__dict__ if hasattr(MODEL,"config") else {},
        prompt=msgs, num_images=n_img)

def sync_gen(msgs:List[Dict], imgs, max_t:int, temp:float, top_p:float, strip:bool):
    prompt=prompt_from_msgs(msgs, len(imgs))
    start=time.time()
    out=generate(MODEL,PROCESSOR,prompt=prompt,image=imgs,
                 max_tokens=max_t,temp=temp,top_p=top_p,verbose=False)
    out=_strip(out, strip)
    dt=time.time()-start
    usage=Usage(input_tokens=len(prompt), output_tokens=len(out),
                total_tokens=len(prompt)+len(out),
                tokens_per_second=(len(prompt)+len(out))/max(dt,1e-6))
    return out, usage

async def stream_chunks(req_id,prompt,imgs,max_t,temp,top_p,strip,created):
    inside=False; seen=0
    async for res in stream_generate(MODEL,PROCESSOR,prompt,image=imgs,
                                     max_tokens=max_t,temp=temp,top_p=top_p):
        ch=res.text
        if strip:
            buf=""; i=0
            while i<len(ch):
                if not inside and ch.startswith("<think>",i): inside=True; i+=7; continue
                if inside and ch.startswith("</think>",i): inside=False; i+=8; continue
                if inside: i+=1; continue
                buf+=ch[i]; seen+=ch[i].strip()!= "" and 1 or 0; i+=1
            ch = buf if seen<4 else ch
        if not ch: continue
        yield f"data: {json.dumps({'id':req_id,'object':'chat.completion.chunk','created':created,'model':MODEL_NAME,'choices':[{'index':0,'delta':{'content':ch},'finish_reason':None}]},ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)

# ── FastAPI ──────────────────────────────────────────────────────
app=FastAPI()

def _strip_flag(body)->bool:
    return bool(getattr(body,"strip_thinking",False)
                or getattr(body,"stripThinking",False)
                or args.strip_thinking)

@app.post("/v1/chat/completions")
async def chat(body:ChatReq):
    if body.model!=MODEL_NAME: raise HTTPException(400,"Unknown model")
    imgs=collect_images(body.images, body.videos); strip=_strip_flag(body)
    if not body.stream:
        txt,use=sync_gen([m.model_dump() for m in body.messages],imgs,
                         body.max_tokens,body.temperature,body.top_p,strip)
        return {"id":f"chatcmpl-{uuid.uuid4()}","object":"chat.completion",
                "created":int(time.time()),"model":MODEL_NAME,
                "choices":[{"index":0,"message":{"role":"assistant","content":txt},
                            "finish_reason":"stop"}],"usage":use.model_dump()}
    prompt=prompt_from_msgs([m.model_dump() for m in body.messages],len(imgs))
    rid=f"chatcmpl-{uuid.uuid4()}"; created=int(time.time())
    return StreamingResponse(stream_chunks(rid,prompt,imgs,
                                           body.max_tokens,body.temperature,body.top_p,
                                           strip,created),
                             media_type="text/event-stream")

@app.post("/v1/responses")
async def responses(body:RespReq):
    if body.model!=MODEL_NAME: raise HTTPException(400,"Unknown model")
    hist=STATE.get(body.previous_response_id,[])
    hist += [{"role":"user","content":body.input}] if isinstance(body.input,str) else \
            [{"role":m.get('role','user'),"content":m.get('content','')} for m in body.input]
    imgs=collect_images(body.images, body.videos); strip=_strip_flag(body)
    max_t=body.max_output_tokens or MAX_NEW_TOKENS
    if not body.stream:
        txt,use=sync_gen(hist,imgs,max_t,body.temperature,body.top_p,strip)
        rid=f"resp_{uuid.uuid4().hex}"; STATE[rid]=hist+[{"role":"assistant","content":txt}]
        return {"id":rid,"object":"response","created_at":time.time(),"model":MODEL_NAME,
                "output":[{"id":"msg_"+uuid.uuid4().hex,"type":"message","role":"assistant",
                           "content":[{"type":"output_text","text":txt}]}],
                "output_text":txt,"temperature":body.temperature,"top_p":body.top_p,
                "previous_response_id":body.previous_response_id,
                "status":"completed","usage":use.model_dump()}
    prompt=prompt_from_msgs(hist,len(imgs)); rid=f"resp_{uuid.uuid4().hex}"; created=int(time.time())
    async def gen():
        async for c in stream_chunks(rid,prompt,imgs,max_t,
                                     body.temperature,body.top_p,strip,created):
            yield c
        STATE[rid]=hist; yield "data: [DONE]\n\n"
    return StreamingResponse(gen(),media_type="text/event-stream")

# ── main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    LOG.info("Serving %s on %s:%d", MODEL_NAME, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)