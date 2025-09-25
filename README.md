# VLM Explorer

Modern PySide6 desktop app for interacting with Vision-Language Models (VLMs). Load one or two images, ask questions in a chat-style UI, and get answers from a wide range of pre-configured models or a local model directory.

## Key Features

- **Chat-style UI**: Conversational layout with message bubbles and timestamps.
- **Model-aware config**: Settings adapt to the selected model (tasks, quantization hints, device mapping).
- **Model preloading**: Load/unload models in the background for faster inference.
- **Two-screen support**: Optionally combine two images vertically for NDS-style inputs.
- **Export conversation**: Save chat + metadata as structured JSON.
- **Built-in FastAPI server**: `/health` and `/infer` endpoints for external integrations.
- **Modern dark theme**: Styled with a sleek, readable interface.

## Prerequisites

- Python 3.8+
- Windows/macOS/Linux
- CUDA-enabled GPU recommended (CPU supported)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python vlm_gui.py
```

## Using the App

1. **Load Image(s)**: Click “🖼️ Load Image”. For two-screen mode, choose “Two Screens” and load the second image.
2. **Select Model**: Pick from the “Model” dropdown or click “📁 Local” to select a local HF model folder.
3. **Chat**: Type a question in the single-line input and press Enter or click “Send”.
4. **Export**: Click “💾 Export” to save the conversation and metadata as JSON.
5. **Preload/Unload**: Use the “🔄 Load Model” and “⏏ Unload Model” buttons on the Models page.

### Model-Aware Configuration

- The Config page updates options based on the chosen model family.
- LLaVA/OneVision/VQA-style models default to **VQA** task.
- Large models (≥7B) will recommend **4-bit** quantization and may enable **device_map=auto** when multi-GPU is available.

## Available Models

See `vlm_gui.py` for the full `available_models` mapping. You can also load a local directory containing standard HF files (e.g. `config.json`, `model.safetensors`, tokenizer files).

## FastAPI Server

The desktop app can start a FastAPI server for external integrations.

Endpoints (`app/server.py`):

- `GET /health` → Returns `{status, cuda, vram}`.
- `POST /infer` → Request body:

```json
{
  "model_name": "llava-hf/llava-v1.6-mistral-7b-hf",
  "task": "VQA",
  "prompt": "What is happening?",
  "cache_dir": null,
  "gen_params": {"answer_top_k": 1, "temperature": 1.0, "top_p": 0.9, "num_beams": 1, "max_new_tokens": 64},
  "preprocess": {"use_fp16": true, "max_image_size": 0},
  "load_opts": {"use_8bit": false, "use_4bit": false, "device_map_auto": false},
  "image_b64_top": "<base64>",
  "image_b64_bottom": null
}
```

Response:

```json
{ "answer": "...", "elapsed_seconds": 0.73 }
```

## Export Format

The “Export” button writes a JSON object containing:

```json
{
  "model": "<model id or local path>",
  "screens": "One Screen | Two Screens",
  "image": "<path>",
  "image2": "<path or null>",
  "gen_params": {
    "answer_top_k": 1,
    "temperature": 1.0,
    "top_p": 0.9,
    "num_beams": 1,
    "max_new_tokens": 64
  },
  "messages": [
    {"role": "user", "content": "...", "timestamp": "HH:MM"},
    {"role": "assistant", "content": "...", "timestamp": "HH:MM"}
  ],
  "last_answer": "..."
}
```

## Developer Guide

- **Code style**: PEP8 + Ruff. Run `ruff check . --fix` before committing.
- **Virtualenv**: Use `venv/` (ignored by `.gitignore`).
- **Logging**: See `vlm_explorer.log` (ignored by `.gitignore`).
- **Server**: Code in `app/server.py`. Worker logic in `app/vlm_worker.py`.
- **Tests**: Place under `tests/`.

### Common Issues

- If VRAM is low, the app will warn you before inference. Consider enabling quantization.
- For local models, ensure the directory contains standard HF files and that you have permissions to read them.

## License

MIT
