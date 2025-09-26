"""Lightweight routing tests for app.vlm_worker.run_vlm_inference.

These tests monkeypatch the pipeline factory inside app.vlm_worker to verify
that the function sends the correct arguments for chat-like vs captioning models
without requiring model downloads or an active Qt event loop.

Run as a script: python tests/test_vlm_routing.py
"""
from __future__ import annotations

from typing import Any, Dict
from PIL import Image

import app.vlm_worker as vlm_worker


class DummyPipeline:
    def __init__(self, task: str | None):
        self.task = task or "image-to-text"
        self.calls: list[Dict[str, Any]] = []

    def __call__(self, image, **kwargs):  # noqa: D401
        # record the call for assertions
        self.calls.append({"image": image, "kwargs": dict(kwargs)})
        # emulate transformers pipelines: return a list of dicts with text
        return [{"generated_text": "OK"}]


# simple shared holder for last returned pipeline instance
_LAST_INSTANCE: dict[str, Any] = {}


def fake_pipeline_factory(**kwargs):
    inst = DummyPipeline(task=kwargs.get("task"))
    _LAST_INSTANCE["inst"] = inst
    return inst


def _make_test_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(0, 0, 0))


def test_chat_like_non_preloaded() -> bool:
    """Chat-like models should pass a structured conversation object to text=.

    We simulate a local path so run_vlm_inference sets task to image-text-to-text.
    """
    saved_pipeline = vlm_worker.pipeline
    try:
        vlm_worker.pipeline = fake_pipeline_factory  # monkeypatch
        img = _make_test_image()
        answer, _ = vlm_worker.run_vlm_inference(
            model_name=r"C:\\fake\\Qwen2.5-VL-7B-Instruct",  # local path heuristics
            prompt="What do you see?",
            cache_dir=None,
            gen_params={"max_new_tokens": 8},
            preprocess={"use_fp16": False},
            pil_image=img,
            image_path=None,
            load_opts={"use_4bit": False, "use_8bit": False, "device_map_auto": False},
            task="VQA",
        )
        inst = _LAST_INSTANCE.get("inst")
        assert isinstance(inst, DummyPipeline), "pipeline factory not used"
        assert inst.task and "image-text-to-text" in inst.task, f"unexpected task: {inst.task}"
        assert answer == "OK", f"unexpected answer normalization: {answer!r}"
        assert inst.calls, "pipeline was never called"
        call = inst.calls[-1]["kwargs"]
        assert "text" in call, "chat-like should pass text parameter"
        text_val = call["text"]
        assert isinstance(text_val, list) and text_val and isinstance(text_val[0], dict), (
            "expected structured conversation object for text="
        )
        return True
    finally:
        vlm_worker.pipeline = saved_pipeline


def test_captioning_non_preloaded() -> bool:
    """Captioning models should not receive text=; image only."""
    saved_pipeline = vlm_worker.pipeline
    try:
        vlm_worker.pipeline = fake_pipeline_factory  # monkeypatch
        img = _make_test_image()
        answer, _ = vlm_worker.run_vlm_inference(
            model_name=r"C:\\fake\\caption-model",  # treated as local path; not chat-like
            prompt="Describe the image.",
            cache_dir=None,
            gen_params={},
            preprocess={"use_fp16": False},
            pil_image=img,
            image_path=None,
            load_opts={"use_4bit": False, "use_8bit": False, "device_map_auto": False},
            task="Image-to-Text",
        )
        inst = _LAST_INSTANCE.get("inst")
        assert isinstance(inst, DummyPipeline), "pipeline factory not used"
        # For non chat-like local model, task should be image-to-text
        assert inst.task and "image-to-text" in inst.task and "image-text-to-text" not in inst.task, (
            f"unexpected task: {inst.task}"
        )
        assert answer == "OK", f"unexpected answer normalization: {answer!r}"
        assert inst.calls, "pipeline was never called"
        call = inst.calls[-1]["kwargs"]
        assert "text" not in call, "captioning should not pass text parameter"
        return True
    finally:
        vlm_worker.pipeline = saved_pipeline


def main():
    print("🧪 VLM Routing Tests")
    print("=" * 32)
    ok1 = ok2 = False
    try:
        ok1 = test_chat_like_non_preloaded()
        print("✅ chat-like non-preloaded routing test passed")
    except AssertionError as e:
        print(f"❌ chat-like routing FAILED: {e}")
    try:
        ok2 = test_captioning_non_preloaded()
        print("✅ captioning non-preloaded routing test passed")
    except AssertionError as e:
        print(f"❌ captioning routing FAILED: {e}")
    print("\nResult:")
    if ok1 and ok2:
        print("✨ All routing tests passed")
    else:
        print("⚠️ Some routing tests failed")


if __name__ == "__main__":
    main()
