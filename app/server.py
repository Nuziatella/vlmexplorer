"""FastAPI server exposing VLM inference.

Endpoints:
- GET /health               -> server status and CUDA/VRAM info
- POST /infer               -> run inference with one or two images (base64), returns answer and elapsed time

This reuses the shared inference function from app.vlm_worker.run_vlm_inference
so the behavior matches the desktop GUI.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, Optional
import time
import threading

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from .vlm_worker import run_vlm_inference


class GenParams(BaseModel):
    answer_top_k: int = 1
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_beams: Optional[int] = None
    max_new_tokens: Optional[int] = None


class PreprocessOpts(BaseModel):
    use_fp16: bool = True
    max_image_size: int = 0


class LoadOpts(BaseModel):
    use_8bit: bool = False
    use_4bit: bool = False
    device_map_auto: bool = False


class InferRequest(BaseModel):
    model_name: str = Field(..., description="HF model id or local directory path")
    task: str = Field("VQA", description="Either 'VQA' or 'Image-to-Text'")
    prompt: str
    cache_dir: Optional[str] = None
    gen_params: GenParams = Field(default_factory=GenParams)
    preprocess: PreprocessOpts = Field(default_factory=PreprocessOpts)
    load_opts: LoadOpts = Field(default_factory=LoadOpts)
    image_b64_top: str = Field(..., description="Base64-encoded PNG/JPEG (top or single image)")
    image_b64_bottom: Optional[str] = Field(None, description="Optional second image (NDS bottom)")


class InferResponse(BaseModel):
    answer: str
    elapsed_seconds: float


app = FastAPI(title="VLM Explorer API")


def _decode_image(b64_str: str) -> Image.Image:
    try:
        data = base64.b64decode(b64_str)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        return im
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {exc}") from exc


@app.get("/health")
async def health() -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    vram = None
    if cuda:
        try:
            free, total = torch.cuda.mem_get_info()
            vram = {
                "free_bytes": int(free),
                "total_bytes": int(total),
                "free_gb": round(free / (1024**3), 3),
                "total_gb": round(total / (1024**3), 3),
                "device": torch.cuda.get_device_name(0),
            }
        except Exception:
            vram = {"error": "unavailable"}
    # Include a minimal metrics summary for convenience
    m = _get_metrics_snapshot()
    return {
        "status": "ok",
        "cuda": cuda,
        "vram": vram,
        "metrics": {
            "incoming": m["incoming"],
            "succeeded": m["succeeded"],
            "failed": m["failed"],
            "avg_latency_ms": m["avg_latency_ms"],
            "last_latency_ms": m["last_latency_ms"],
            "uptime_seconds": m["uptime_seconds"],
        },
    }


# ---------------------------
# Metrics Implementation
# ---------------------------
_METRICS = {
    "start_time": time.time(),
    "incoming": 0,
    "succeeded": 0,
    "failed": 0,
    "total_latency_ms": 0.0,
    "last_latency_ms": None,
}
_METRICS_LOCK = threading.Lock()


def _get_metrics_snapshot() -> Dict[str, Any]:
    """Return a thread-safe snapshot of server metrics."""
    with _METRICS_LOCK:
        incoming = int(_METRICS["incoming"])
        succeeded = int(_METRICS["succeeded"])
        failed = int(_METRICS["failed"])
        total_latency_ms = float(_METRICS["total_latency_ms"])  # total for successes
        last_latency_ms = (
            float(_METRICS["last_latency_ms"]) if _METRICS["last_latency_ms"] is not None else None
        )
        uptime_seconds = float(max(0.0, time.time() - _METRICS["start_time"]))
    avg_latency_ms = (total_latency_ms / succeeded) if succeeded > 0 else 0.0
    return {
        "incoming": incoming,
        "succeeded": succeeded,
        "failed": failed,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "last_latency_ms": round(last_latency_ms, 3) if last_latency_ms is not None else None,
        "uptime_seconds": round(uptime_seconds, 3),
        "start_time": _METRICS["start_time"],
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Expose server metrics: counters and latency stats."""
    return _get_metrics_snapshot()


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    # Compose PIL image: stack vertically if two images
    im_top = _decode_image(req.image_b64_top)
    pil_image: Optional[Image.Image]
    if req.image_b64_bottom:
        im_bottom = _decode_image(req.image_b64_bottom)
        w = max(im_top.width, im_bottom.width)
        h = im_top.height + im_bottom.height
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(im_top, (0, 0))
        canvas.paste(im_bottom, (0, im_top.height))
        pil_image = canvas
    else:
        pil_image = im_top

    # Convert pydantic to plain dicts
    gen_params: Dict[str, Any] = req.gen_params.model_dump(exclude_none=True)
    preprocess: Dict[str, Any] = req.preprocess.model_dump()
    load_opts: Dict[str, Any] = req.load_opts.model_dump()

    # Set HF_HOME if provided
    cache_dir = req.cache_dir or os.environ.get("HF_HOME")

    # Track metrics and measure end-to-end server latency
    with _METRICS_LOCK:
        _METRICS["incoming"] += 1
    server_t0 = time.perf_counter()
    try:
        answer, elapsed = run_vlm_inference(
            model_name=req.model_name,
            prompt=req.prompt,
            cache_dir=cache_dir,
            gen_params=gen_params,
            preprocess=preprocess,
            pil_image=pil_image,
            image_path=None,
            load_opts=load_opts,
            task=req.task,
        )
        server_latency_ms = (time.perf_counter() - server_t0) * 1000.0
        with _METRICS_LOCK:
            _METRICS["succeeded"] += 1
            _METRICS["total_latency_ms"] += float(server_latency_ms)
            _METRICS["last_latency_ms"] = float(server_latency_ms)
        return InferResponse(answer=answer, elapsed_seconds=float(elapsed))
    except Exception:
        server_latency_ms = (time.perf_counter() - server_t0) * 1000.0
        with _METRICS_LOCK:
            _METRICS["failed"] += 1
            _METRICS["last_latency_ms"] = float(server_latency_ms)
        raise
