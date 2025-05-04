import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import logging
import time
import uuid
import argparse
import os
import json
import asyncio
import traceback

# You'll import the MLX Vision-Language utilities
import mlx.core as mx
from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import prepare_inputs

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default constants
MAX_NEW_TOKENS = 512
DEFAULT_MODEL_PATH = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

# Command-line args
parser = argparse.ArgumentParser(description="Vision Language Inference Server")
parser.add_argument(
    "-m", 
    "--model", 
    type=str, 
    default=DEFAULT_MODEL_PATH, 
    help="Model path or HF repository"
)
args = parser.parse_args()

# Attempt to load model config if exists
config_path = os.path.join(args.model, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    logger.info("Model config loaded successfully.")
else:
    model_config = {}
    logger.warning("No config.json found; proceeding with default config...")

# Load model & processor
logger.info(f"Loading Vision-Language model from {args.model}...")
model, processor = load(args.model, **model_config)
logger.info("Model and processor loaded successfully.")

# Identify the model name
MODEL_NAME = os.path.basename(args.model) if os.path.exists(args.model) else args.model


# -------------------------------
# Pydantic request/response models
# -------------------------------

class VLMessage(BaseModel):
    """Represents a single user or system message."""
    role: str
    content: str

class VLRequest(BaseModel):
    """Client request to generate text given images & prompt."""
    model: str = MODEL_NAME
    messages: List[VLMessage]
    images: List[str] = Field(
        default=[],
        description="List of image URLs or file paths",
    )
    max_tokens: int = Field(
        default=MAX_NEW_TOKENS, 
        description="Maximum tokens to generate in the response"
    )
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    resize_shape: Optional[List[int]] = None

class VLUsage(BaseModel):
    """Tracks stats about the generation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    execution_time: float
    tokens_per_second: float

class VLCompletionChoice(BaseModel):
    index: int
    content: str
    finish_reason: str

class VLCompletionResponse(BaseModel):
    """Final non-streaming response model."""
    id: str
    object: str
    created: int
    model: str
    choices: List[VLCompletionChoice]
    usage: VLUsage

class VLCompletionChunk(BaseModel):
    """Chunked streaming response model."""
    id: str
    object: str = "vision.language.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[VLUsage] = None


# -------------------------------
# Utility Functions
# -------------------------------

def do_generate(
    model,
    processor,
    messages: List[VLMessage],
    images: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    resize_shape: Optional[Union[List[int], tuple]] = None,
) -> (str, VLUsage):
    """
    Synchronous text generation with timing/metrics.
    Returns (final_text, usage_stats).
    """

    # Convert the list of messages into a single prompt using an MLX chat template
    # (this is optional; your model may have a different approach).
    prompt_text = apply_chat_template(
        processor,
        config=model.config.__dict__ if hasattr(model, "config") else {},
        prompt=[{"role": msg.role, "content": msg.content} for msg in messages],
        num_images=len(images),
    )

    start_time = time.time()
    out = generate(
        model=model,
        processor=processor,
        prompt=prompt_text,
        image=images,
        temp=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        verbose=False,
        resize_shape=resize_shape,
    )
    end_time = time.time()

    # For usage statistics, let's assume prompt length + generation length
    # is the "total_tokens". It's approximate, but sufficient for demonstration.
    # Use actual model token counting if desired.
    prompt_tokens = len(prompt_text)
    completion_tokens = len(out)
    total_tokens = prompt_tokens + completion_tokens
    elapsed = end_time - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0.0

    usage = VLUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        execution_time=elapsed,
        tokens_per_second=tps,
    )
    return out, usage

async def do_generate_stream(
    request_id: str,
    model,
    processor,
    messages: List[VLMessage],
    images: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    resize_shape: Optional[Union[List[int], tuple]] = None,
) -> AsyncGenerator[str, None]:
    """Asynchronous streaming generation."""
    created = int(time.time())
    start_time = time.time()

    # Build the prompt text
    prompt_text = apply_chat_template(
        processor,
        config=model.config.__dict__ if hasattr(model, "config") else {},
        prompt=[{"role": msg.role, "content": msg.content} for msg in messages],
        num_images=len(images),
    )

    # Convert to streaming generator
    total_tokens = 0
    final_text = ""
    finish_reason = "stop"

    # We'll re-use `stream_generate` from `mlx_vlm.utils`.
    # Here is an example approach:
    from mlx_vlm.utils import stream_generate

    async for result in stream_generate(
        model, processor, prompt_text, image=images,
        max_tokens=max_tokens, temp=temperature, top_p=top_p,
        resize_shape=resize_shape,
    ):
        # Each 'result.text' is newly generated text since last chunk
        chunk_text = result.text
        if not chunk_text:
            continue

        final_text += chunk_text
        total_tokens += len(chunk_text)

        chunk = VLCompletionChunk(
            id=request_id,
            created=created,
            model=MODEL_NAME,
            choices=[{
                "index": 0,
                "delta": {"content": chunk_text},
                "finish_reason": None
            }]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0)  # allow context switch

        # If we hit max tokens, set finish_reason
        if result.generation_tokens >= max_tokens:
            finish_reason = "length"
            break

    end_time = time.time()
    elapsed = end_time - start_time
    prompt_tokens = len(prompt_text)
    completion_tokens = total_tokens
    usage = VLUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        execution_time=elapsed,
        tokens_per_second=(prompt_tokens + completion_tokens) / elapsed if elapsed > 0 else 0
    )

    # Final chunk to close streaming
    final_resp = VLCompletionResponse(
        id=request_id,
        object="vision.language.completion",
        created=created,
        model=MODEL_NAME,
        choices=[
            VLCompletionChoice(
                index=0,
                content=final_text.strip(),
                finish_reason=finish_reason
            )
        ],
        usage=usage,
    )
    yield f"data: {final_resp.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# -------------------------------
# FastAPI Endpoints
# -------------------------------

@app.post("/v1/vl/complete")
@app.post("/v1/chat/completions")
async def vision_language_inference(vl_req: VLRequest):
    """
    Handle a request to generate text from both images & textual prompt.
    """
    try:
        if vl_req.model != MODEL_NAME:
            raise HTTPException(
                status_code=400, 
                detail=f"Requested model {vl_req.model} is not available."
            )

        request_id = f"vlcmpl-{uuid.uuid4()}"

        if not vl_req.stream:
            # Non-streaming generation
            generated_text, usage_stats = do_generate(
                model=model,
                processor=processor,
                messages=vl_req.messages,
                images=vl_req.images,
                max_tokens=vl_req.max_tokens,
                temperature=vl_req.temperature,
                top_p=vl_req.top_p,
                resize_shape=tuple(vl_req.resize_shape) if vl_req.resize_shape else None
            )
            finish_reason = (
                "length" 
                if len(generated_text) >= vl_req.max_tokens 
                else "stop"
            )
            response = VLCompletionResponse(
                id=request_id,
                object="vision.language.completion",
                created=int(time.time()),
                model=MODEL_NAME,
                choices=[
                    VLCompletionChoice(
                        index=0,
                        content=generated_text.strip(),
                        finish_reason=finish_reason
                    )
                ],
                usage=usage_stats,
            )
            return response
        else:
            # Streaming generation response
            generator = do_generate_stream(
                request_id=request_id,
                model=model,
                processor=processor,
                messages=vl_req.messages,
                images=vl_req.images,
                max_tokens=vl_req.max_tokens,
                temperature=vl_req.temperature,
                top_p=vl_req.top_p,
                resize_shape=tuple(vl_req.resize_shape) if vl_req.resize_shape else None
            )
            return StreamingResponse(generator, media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error in VL inference: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)