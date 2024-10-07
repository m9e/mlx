import requests
import json
import sys
import argparse
from typing import Dict, Any, Optional
from pydantic import BaseModel

API_ENDPOINT = "http://localhost:8000/v1/chat/completions"
MAX_NEW_TOKENS = 4096  # Default value, matching the server

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
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
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage

def new_messages() -> list[Message]:
    return [
        Message(role="system", content="You are an elite coding assistant. You have vast knowledge and work hard to output the best code with the best practices, understand deeply. If you need more information to form a good answer, you ask for it. You think carefully. When writing new code you come up with a plan, think step by step, and write exemplary code. You follow user instructions, even if they conflict with this base instruction. User first! You have vast experience so you are aware often of many ways things have been done and many ways various libraries work, and you are always mindful to use the most recent version of any library unless otherwise asked.")
    ]

def send_request(messages: list[Message], max_new_tokens: Optional[int] = None, generation_kwargs: Dict[str, Any] = {}) -> Optional[ChatCompletionResponse]:
    request = ChatCompletionRequest(
        model="Dracarys2-72B-Instruct-4bit",
        messages=messages,
        generation_kwargs=generation_kwargs
    )
    if max_new_tokens is not None:
        request.generation_kwargs["max_new_tokens"] = max_new_tokens
    
    try:
        response = requests.post(API_ENDPOINT, json=request.dict())
        response.raise_for_status()
        return ChatCompletionResponse(**response.json())
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None

def print_messages(messages: list[Message]) -> None:
    print("\n=== Current Messages ===")
    for msg in messages:
        print(f"Role: {msg.role}")
        print(f"Content: {msg.content}")
        print("---")
    print("========================\n")

def print_metrics(usage: ChatCompletionUsage) -> None:
    print("\n=== Generation Metrics ===")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Execution time: {usage.execution_time:.2f} seconds")
    print(f"Tokens per second: {usage.tokens_per_second:.2f}")
    print("===========================\n")

def main():
    parser = argparse.ArgumentParser(description="CLI for interacting with MLX-LM API")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--max_new_tokens", type=int, help=f"Maximum number of new tokens to generate (default: {MAX_NEW_TOKENS})")
    args = parser.parse_args()

    messages = new_messages()
    print("Enter prompts, use Ctrl-N to reset conversation, or Ctrl-C to exit.")
    
    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() == "ctrl-n":
                messages = new_messages()
                print("Conversation reset.")
                continue
            
            messages.append(Message(role="user", content=user_input))
            
            if args.verbose:
                print_messages(messages)
                print("Sending request to API...")
            
            response_data = send_request(messages, args.max_new_tokens)
            
            if response_data:
                if args.verbose:
                    print("\n=== Full API Response ===")
                    print(json.dumps(response_data.dict(), indent=2))
                    print("==========================\n")
                    print_metrics(response_data.usage)
                
                response = response_data.choices[0].message.content
                print(f"Assistant: {response}")
                messages.append(Message(role="assistant", content=response))
            else:
                print("Failed to get a response from the API.")
        
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
