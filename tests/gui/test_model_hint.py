import os
import sys

# Ensure offscreen Qt for headless testing
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.vlm_gui import VLMApp  # noqa: E402


def test_model_hint_updates_with_model_selection():
    app = QApplication.instance() or QApplication([])
    win = VLMApp()

    # Ensure hint label exists
    assert hasattr(win, "model_hint_label")

    # Pick a LLaVA model to trigger VQA hint text
    target_model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    idx = None
    for i in range(win.model_combo.count()):
        if win.model_combo.itemData(i) == target_model_id:
            idx = i
            break
    assert idx is not None, "Expected LLaVA model not found in model_combo"

    win.model_combo.setCurrentIndex(idx)
    app.processEvents()

    # Force hint update and verify content
    win.update_model_hint()
    hint = win.model_hint_label.text()

    assert target_model_id in hint
    assert "Task: " in hint
    # For llava and VQA default, we expect a direct question hint
    assert "Hint" in hint

    # Cleanup
    win.close()
    app.quit()
