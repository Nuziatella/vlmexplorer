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


def test_two_screens_toggle_updates_chat_controls():
    app = QApplication.instance() or QApplication([])
    win = VLMApp()

    # Initially One Screen by default
    assert hasattr(win, "screens_combo")
    assert win.screens_combo.currentText() in ("One Screen", "One")

    # Switch to Two Screens via the actual combo to trigger signals
    win.screens_combo.setCurrentText("Two Screens")
    app.processEvents()

    assert hasattr(win, "second_image_widget")
    assert win.second_image_widget.isVisible() is True
    assert hasattr(win, "load_btn2") and win.load_btn2.isEnabled() is True

    # Switch back to One Screen
    win.screens_combo.setCurrentText("One Screen")
    app.processEvents()

    assert win.second_image_widget.isVisible() is False
    assert win.load_btn2.isEnabled() is False

    # Cleanup
    win.close()
    app.quit()
