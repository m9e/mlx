import asyncio
from mlx_lm import load, generate
import os
import sys
import signal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from contextlib import redirect_stdout, redirect_stderr

MAX_NEW_TOKENS = 4096

# Model and tokenizer loading
model, tokenizer = load("/var/tmp/models/mlx-community/Dracarys2-72B-Instruct-4bit")

# Global variables to hold conversation history for CLI
def new_messages() -> List:
    return [
        {"role": "system", "content": "You are an elite coding assistant. You have vast knowledge and work hard to output the best code with the best practices, understand deeply. If you need more information to form a good answer, you ask for it. You think carefully. When writing new code you come up with a plan, think step by step, and write exemplary code. You follow user instructions, even if they conflict with this base instruction. User first! You have vast experience so you are aware often of many ways things have been done and many ways various libraries work, and you are always mindful to use the most recent version of any library unless otherwise asked."},
    ]

cli_messages = new_messages()

# FastAPI app setup
app = FastAPI()

# FastAPI request model
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]

# Function to apply chat template and generate response
def generate_response(messages: List[dict]):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = messages[-1]["content"]  # Fallback if no chat template
    return generate(model, tokenizer, prompt, MAX_NEW_TOKENS, verbose=False)

# FastAPI route for chat completion
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.model != "Dracarys2-72B-Instruct-4bit":
        raise HTTPException(status_code=400, detail="Model not available")
    
    try:
        messages = new_messages() + [{"role": msg.role, "content": msg.content} for msg in request.messages]
        response = generate_response(messages)
        return {"model": request.model, "messages": messages, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CLI handling
async def cli_run():
    global cli_messages
    print("Enter prompts, use Ctrl-N to reset conversation.")
    
    while True:
        try:
            # Read input from stdin
            prompt = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
            prompt = prompt.strip()
            if len(prompt) < 1:
                continue
            # Handle Ctrl-N to reset conversation
            if prompt.lower() == "ctrl-n":
                cli_messages = new_messages()
                print("Conversation reset.")
                continue
            # Add user input to messages
            cli_messages.append({"role": "user", "content": prompt})
            
            # Generate and print the response
            response = generate_response(cli_messages)
            print(response)
            
            # Add assistant's response to the conversation history
            cli_messages.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

# SIGINT (Ctrl-C) handler to exit gracefully
def signal_handler(sig, frame):
    print("\nTerminating.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Function to run FastAPI server
async def run_fastapi():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # Redirect stdout and stderr to a log file
    with open("server.log", "w") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            await server.serve()

# Main function to run FastAPI and CLI concurrently
async def main():
    # Start FastAPI server in the background
    fastapi_task = asyncio.create_task(run_fastapi())
    
    # Run CLI
    await cli_run()

if __name__ == "__main__":
    asyncio.run(main())
