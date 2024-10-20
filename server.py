import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator
from mlx_lm.utils import generate_step, load
import logging
import time
import uuid
import argparse
import os
import json
import mlx.core as mx
import asyncio
import traceback

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_NEW_TOKENS = 8192
DEFAULT_MODEL_PATH = "/var/tmp/models/mlx-community/Dracarys2-72B-Instruct-4bit"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="MLX-LM API Server")
parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL_PATH, help="Model path or Hugging Face model name")
args = parser.parse_args()

# Load model and tokenizer
logger.info(f"Loading model from {args.model}...")

# load configs
config_path = os.path.join(DEFAULT_MODEL_PATH, "config.json")
tokenizer_config_path = os.path.join(DEFAULT_MODEL_PATH, "tokenizer_config.json")
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
MODEL_NAME = os.path.basename(args.model) if os.path.exists(args.model) else args.model

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=MAX_NEW_TOKENS)
    temperature: float = 1.0
    top_p: float = 1.0
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

def is_stop_token(token_id):
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    return tokenizer.decode([token_id]) in stop_tokens

def generate_with_metrics(model, tokenizer, prompt, max_new_tokens, **kwargs):
    input_tokens = len(prompt)
    start_time = time.time()
    response = ""
    output_tokens = 0
    for token, _ in generate_step(prompt=prompt, model=model, **kwargs):
        if is_stop_token(token):
            break
        response += tokenizer.decode([token])
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

async def generate_stream(prompt: mx.array, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    tokens = []
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4()}"
    start_time = time.time()
    finish_reason = "stop"
    full_response = ""

    for token, _ in generate_step(
        prompt=prompt,
        model=model,
        temp=request.temperature,
        top_p=request.top_p
    ):
        if is_stop_token(token):
            finish_reason = "stop"
            break
        tokens.append(token)
        new_text = tokenizer.decode([token])
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
    prompt_tokens = len(prompt)
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
        
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            prompt = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt = tokenizer.encode(prompt)

        prompt = mx.array(prompt)

        if request.stream:
            return StreamingResponse(generate_stream(prompt, request), media_type="text/event-stream")
        
        full_response, metrics = generate_with_metrics(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temp=request.temperature,
            top_p=request.top_p
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
