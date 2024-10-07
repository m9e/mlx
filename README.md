# MLX-LM API Server and Client

This project provides a FastAPI-based server (`server.py`) for serving MLX-LM models and a command-line client (`infer.py`) for interacting with the API.

Dead simple but was just building something to wrap around mlx while driving and experimenting with dracarys-72b locally

## Server (server.py)

The server script sets up a FastAPI application that serves MLX-LM models for chat completions.

### Features:
- Loads and serves MLX-LM models
- Provides a `/v1/chat/completions` endpoint compatible with OpenAI's API
- Supports custom generation parameters
- Includes metrics for token usage and generation speed

### Usage:

To run the server:

```
python server.py -m /path/to/your/model
```

Optional arguments:
- `-m` or `--model`: Specify the path to the MLX-LM model (default: "/var/tmp/models/mlx-community/Dracarys2-72B-Instruct-4bit")

The server will start on `http://0.0.0.0:8000` by default.

## Client (infer.py)

The client script provides a command-line interface for interacting with the MLX-LM API server.

### Features:
- Interactive chat-like interface
- Supports conversation reset (Ctrl-N)
- Verbose mode for detailed API responses
- Customizable maximum token generation

### Usage:

To run the client:

```
python infer.py [options]
```

Optional arguments:
- `-v` or `--verbose`: Enable verbose mode to see detailed API responses and metrics
- `--max_new_tokens`: Set the maximum number of new tokens to generate (default: 4096)

### Interacting with the Client:
- Enter your prompts at the `>` prompt
- Use `Ctrl-N` to reset the conversation
- Use `Ctrl-C` to exit the program

## API Endpoint

The server provides a single endpoint for chat completions:

- **URL**: `/v1/chat/completions`
- **Method**: POST
- **Request Body**: JSON object with the following structure:
  ```json
  {
    "model": "model_name",
    "messages": [
      {"role": "user", "content": "Your message here"}
    ],
    "generation_kwargs": {}
  }
  ```
- **Response**: JSON object containing the generated response and usage statistics

For more details on the API structure and usage, refer to the `ChatCompletionRequest` and `ChatCompletionResponse` classes in the server code.
