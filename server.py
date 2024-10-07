import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from mlx_lm import load, generate
import logging
import time
import uuid

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_NEW_TOKENS = 4096

# Load model and tokenizer
logger.info("Loading model and tokenizer...")
model, tokenizer = load("/var/tmp/models/mlx-community/Dracarys2-72B-Instruct-4bit")
logger.info("Model and tokenizer loaded successfully.")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    generation_kwargs: Dict[str, Any] = {}

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
    object: str = "chat.completion"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def generate_with_metrics(model, tokenizer, prompt, max_new_tokens, **kwargs):
    # Count input tokens
    input_tokens = count_tokens(prompt, tokenizer)
    
    # Start timing
    start_time = time.time()
    
    # Generate response
    response = generate(model, tokenizer, prompt, max_new_tokens, **kwargs)
    
    # End timing
    end_time = time.time()
    
    # Count output tokens
    output_tokens = count_tokens(response, tokenizer)
    
    # Calculate metrics
    total_tokens = input_tokens + output_tokens
    execution_time = end_time - start_time
    tokens_per_second = total_tokens / execution_time if execution_time > 0 else 0
    
    metrics = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "execution_time": execution_time,
        "tokens_per_second": tokens_per_second
    }
    
    return response, metrics

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.model != "Dracarys2-72B-Instruct-4bit":
        raise HTTPException(status_code=400, detail="Model not available")
    
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = messages[-1]["content"]
        
        # Extract max_new_tokens from generation_kwargs or use default
        max_new_tokens = request.generation_kwargs.get('max_new_tokens', MAX_NEW_TOKENS)
        
        # Remove max_new_tokens from generation_kwargs if present
        generation_kwargs = {k: v for k, v in request.generation_kwargs.items() if k != 'max_new_tokens' and k in generate.__code__.co_varnames}
        
        # Call generate_with_metrics
        response, metrics = generate_with_metrics(model, tokenizer, prompt, max_new_tokens, **generation_kwargs)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=response
                    ),
                    finish_reason="stop"  # Assuming it always stops normally
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=metrics["prompt_tokens"],
                completion_tokens=metrics["completion_tokens"],
                total_tokens=metrics["total_tokens"],
                execution_time=metrics["execution_time"],
                tokens_per_second=metrics["tokens_per_second"]
            )
        )
    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
