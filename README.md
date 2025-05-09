# MLX-LM 🦙 — Drop-in OpenAI-style API for any local MLX model

A FastAPI micro-server (server.py) that speaks the OpenAI
`/v1/chat/completions` dialect, plus a tiny CLI client
(`infer.py`) for quick experiments.
Ideal for poking at huge models like Dracarys-72B on an
M4-Max/Studio, hacking on prompts, or piping the output straight into
other tools that already understand the OpenAI schema.

---

## ✨ Highlight reel

| Feature | Details |
|---------|---------|
| 🔌 OpenAI compatible | Same request / response JSON (streaming too) – just change the base-URL. |
| 📦 Zero-config | Point at a local folder or HuggingFace repo (`-m /path/to/model`). |
| 🖼️ Vision-ready | Accepts `{"type":"image_url", …}` parts & base64 URLs – works with Qwen-VL & friends. |
| 🎥 Video-aware | Auto-extracts N key-frames with ffmpeg and feeds them as images. |
| 🧮 Usage metrics | Prompt / completion tokens + tokens-per-second in every response. |
| ⚙️ CLI playground | `infer.py` gives you a REPL with reset (Ctrl-N), verbose mode, max-token flag… |

---

## 🚀 Running the server

```bash
# minimal
python server.py -m /var/tmp/models/mlx-community/Dracarys2-72B-Instruct-4bit

# custom port / host
python server.py -m ./Qwen2.5-VL-72B-Instruct-6bit --host 0.0.0.0 --port 12345
```

Default host/port: `0.0.0.0:18000`

### Most useful flags:

| Flag | Default | What it does |
|------|---------|--------------|
| `-m / --model` | `mlx-community/Qwen2-VL-2B-Instruct-4bit` | Path or HF repo. |
| `--strip-thinking` | off | Removes `<think>…</think>` blocks from model output. |
| `--default-frames` | 4 | How many frames to grab per video URL. |
| `--col-a / --col-b` | qwen / vl | Tokens that trigger the special Qwen-VL vision pipeline. |

---

## 💬 Talking to it with the CLI

```bash
python infer.py --base-url http://localhost:18000/v1 -v --max_new_tokens 2048
```

### Interactive keys
- Ctrl-N: reset conversation
- Ctrl-C: quit

---

## 🌐 HTTP API

POST `/v1/chat/completions`

```json
{
  "model": "Dracarys2-72B-Instruct-4bit",
  "messages": [
    { "role": "user",
      "content": [
        { "type": "text", "text": "Describe this image." },
        { "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,..." } }
      ]
    }
  ],
  "max_tokens": 512,
  "stream": false
}
```

Response (truncated):

```json
{
  "id": "chatcmpl-d4c5…",
  "object": "chat.completion",
  "created": 1715242800,
  "model": "Dracarys2-72B-Instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "The image shows…" },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 143,
    "completion_tokens": 87,
    "total_tokens": 230,
    "tokens_per_second": 32.1
  }
}
```

Add `"stream": true` and you'll get Server-Sent Events chunks followed by
`data: [DONE]`.

---

## 🛠️ Internals (two-sentence tour)

* **server.py** – loads the model with mlx-vlm, converts incoming
OpenAI vision messages to the model's chat-template, handles images /
video frames, and streams tokens back.
* **infer.py** – lightweight REPL that keeps conversation context and
shows latency / TPS stats.

That's it – drop it in front of any MLX model and start chatting!
