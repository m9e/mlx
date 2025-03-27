import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
import logging
import time
import uuid
import argparse
import os
import json
import mlx.core as mx
import asyncio
import traceback
from dataclasses import dataclass

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_NEW_TOKENS = 8192
DEFAULT_MODEL_PATH = "/var/tmp/models/mlx-community/DeepSeek-V3-0324-4bit"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="MLX-LM API Server")
parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL_PATH, help="Model path or Hugging Face model name")
parser.add_argument("--max-kv-size", type=int, default=None, help="Maximum KV cache size (optional)")
args = parser.parse_args()

# Load model and tokenizer
logger.info(f"Loading model from {args.model}...")

# Load configs
config_path = os.path.join(DEFAULT_MODEL_PATH, "config.json")
tokenizer_config_path = os.path.join(DEFAULT_MODEL_PATH, "tokenizer_config.json")
model_config = None
tokenizer_config = None

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    logger.info("Model config loaded successfully.")
else:
    logger.warning("Model config not found.")

# Load tokenizer config if available
if os.path.exists(tokenizer_config_path):
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)
    logger.info("Tokenizer config loaded successfully.")
else:
    logger.warning("Tokenizer config not found.")

model, tokenizer = load(args.model, model_config=model_config, tokenizer_config=tokenizer_config)
logger.info("Model and tokenizer loaded successfully.")

# Extract model name from path or use the full path if it's a Hugging Face model
MODEL_NAME = args.model
MODEL_NAME = MODEL_NAME.rstrip('/')
MODEL_NAME = os.path.basename(MODEL_NAME) if os.path.exists(MODEL_NAME) else MODEL_NAME

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=MAX_NEW_TOKENS)
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = 20
    max_kv_size: Optional[int] = None
    stream: bool = False

class ChatCompletionMessage(BaseModel):
    role: str
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: Optional[Any] = None
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    execution_time: float
    tokens_per_second: float

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[ChatCompletionUsage] = None

def is_stop_token(token_str):
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    return token_str in stop_tokens

def generate_with_metrics(model, tokenizer, prompt, max_new_tokens, **kwargs):
    input_tokens = len(prompt) if isinstance(prompt, list) else len(tokenizer.encode(prompt))
    start_time = time.time()
    
    # Create sampler and logits processors based on kwargs
    sampler = make_sampler(
        temp=kwargs.get("temperature", 1.0),
        top_p=kwargs.get("top_p", 1.0)
    )
    
    logits_processors = make_logits_processors(
        logit_bias=None,  # Not implemented in the request model
        repetition_penalty=kwargs.get("repetition_penalty"),
        repetition_context_size=kwargs.get("repetition_context_size", 20)
    )
    
    response = ""
    output_tokens = 0
    
    for gen_output in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        max_kv_size=kwargs.get("max_kv_size", args.max_kv_size)
    ):
        if is_stop_token(gen_output.text):
            break
        response += gen_output.text
        output_tokens += 1
        if output_tokens >= max_new_tokens:
            break
    
    end_time = time.time()
    total_tokens = input_tokens + output_tokens
    execution_time = end_time - start_time
    tokens_per_second = total_tokens / execution_time if execution_time > 0 else 0
    
    metrics = ChatCompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        execution_time=execution_time,
        tokens_per_second=tokens_per_second
    )
    
    return response, metrics

async def generate_stream(prompt, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    tokens = []
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4()}"
    start_time = time.time()
    finish_reason = "stop"
    full_response = ""
    
    # Create sampler and logits processors
    sampler = make_sampler(
        temp=request.temperature,
        top_p=request.top_p
    )
    
    logits_processors = make_logits_processors(
        logit_bias=None,  # Not implemented in the request model
        repetition_penalty=request.repetition_penalty,
        repetition_context_size=request.repetition_context_size
    )
    
    for gen_output in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=request.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        max_kv_size=request.max_kv_size or args.max_kv_size
    ):
        if is_stop_token(gen_output.text):
            finish_reason = "stop"
            break
            
        tokens.append(gen_output.token)
        new_text = gen_output.text
        full_response += new_text
        
        chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": {"content": new_text},
                    "finish_reason": None
                }
            ]
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        
        if len(tokens) >= request.max_tokens:
            finish_reason = "length"
            break
        
        await asyncio.sleep(0)

    end_time = time.time()
    execution_time = end_time - start_time
    prompt_tokens = len(prompt) if isinstance(prompt, list) else len(tokenizer.encode(prompt))
    completion_tokens = len(tokens)
    total_tokens = prompt_tokens + completion_tokens
    tokens_per_second = total_tokens / execution_time if execution_time > 0 else 0

    usage_info = ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        execution_time=execution_time,
        tokens_per_second=tokens_per_second
    )

    final_response = ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=full_response.strip()),
                finish_reason=finish_reason
            )
        ],
        usage=usage_info
    )
    yield f"data: {json.dumps(final_response.model_dump())}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        if request.model != MODEL_NAME:
            raise HTTPException(status_code=400, detail="Model not available")

        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Handle chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        if request.stream:
            return StreamingResponse(generate_stream(prompt, request), media_type="text/event-stream")
        
        # Non-streaming response
        full_response, metrics = generate_with_metrics(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
            max_kv_size=request.max_kv_size
        )

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=full_response.strip()),
                    finish_reason="length" if len(full_response) >= request.max_tokens else "stop"
                )
            ],
            usage=metrics
        )

        return response
    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)