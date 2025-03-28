import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator
from mlx_lm import load, stream_generate
from mlx_lm.utils import generate_step
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
MAX_NEW_TOKENS = 32768
#DEFAULT_MODEL_PATH = "/var/tmp/models/mlx-community/QwQ-32B-8bit"
DEFAULT_MODEL_PATH = "/var/tmp/models/mlx-community/DeepSeek-V3-0324-4bit"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="MLX-LM API Server")
parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL_PATH, help="Model path or Hugging Face model name")
args = parser.parse_args()

# Load model and tokenizer
logger.info(f"Loading model from {args.model}...")

model, tokenizer = load(args.model)
logger.info("Model and tokenizer loaded successfully.")

# Extract model name
MODEL_NAME = os.path.basename(args.model.rstrip('/'))

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

def generate_with_metrics(model, tokenizer, prompt, max_new_tokens):
    start_time = time.time()
    response_output = ""
    output_tokens = 0
    
    # we are duplicating the effort to count
    input_tokens = len(prompt)

    logger.info(f"Starting generation with {input_tokens} input tokens")
    
    # Let's try with only the required parameters
    for response in stream_generate(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=max_new_tokens):
        response_output += response
        output_tokens += 1

        # circuit break  -should not happen
        if output_tokens >= max_new_tokens:
            break
    
    end_time = time.time()
    total_tokens = input_tokens + output_tokens
    execution_time = end_time - start_time
    tokens_per_second = total_tokens / execution_time if execution_time > 0 else 0
    
    logger.info(f"Generation completed: {output_tokens} tokens in {execution_time:.2f}s ({tokens_per_second:.2f} tokens/sec)")
    
    metrics = ChatCompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        execution_time=execution_time,
        tokens_per_second=tokens_per_second
    )
    
    return response, metrics

async def generate_stream(prompt: mx.array, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    start_time = time.time()
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4()}"
    output_tokens = 0
    full_response = ""
    finish_reason = "stop"
    
    
    # Get actual input token count
    input_tokens = len(prompt)

    logger.info(f"Starting stream generation with {input_tokens} input tokens")
    
    for response in stream_generate(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=request.max_tokens):
        full_response += response
        output_tokens += 1
        
        chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": {"content": response},
                    "finish_reason": None
                }
            ]
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        
        # Circuit break if we reach max tokens
        if output_tokens >= request.max_tokens:
            finish_reason = "length"
            break
        
        await asyncio.sleep(0)
    
    end_time = time.time()
    total_tokens = input_tokens + output_tokens
    execution_time = end_time - start_time
    tokens_per_second = total_tokens / execution_time if execution_time > 0 else 0
    
    logger.info(f"Generation completed: {output_tokens} tokens in {execution_time:.2f}s ({tokens_per_second:.2f} tokens/sec)")

    usage_info = ChatCompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
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
            logger.warning(f"Requested model '{request.model}' does not match loaded model '{MODEL_NAME}'")
            # Continue anyway, using the loaded model

        # Format messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Prepare prompt
        logger.info("Preparing prompt")
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            # Modern way - make sure tokenize=True
            logger.info("Using chat template")
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # If prompt is not an array, convert it
        else:
            # Fallback
            logger.info("Using fallback prompt formatting")
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        prompt = tokenizer.encode(prompt_text)
        if not isinstance(prompt, mx.array):
            prompt = mx.array(prompt)
        
        logger.info(f"Prompt prepared - {len(prompt)} tokens")
        
        # Check streaming mode
        if request.stream:
            logger.info("Streaming response")
            return StreamingResponse(
                generate_stream(prompt, request), 
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        logger.info("Generating non-streaming response")
        full_response, metrics = generate_with_metrics(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=request.max_tokens
        )

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=MODEL_NAME,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=full_response.strip()),
                    finish_reason="length" if len(full_response.split()) >= request.max_tokens else "stop"
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