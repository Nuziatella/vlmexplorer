"""Simple smoke test for the VLM Explorer GUI.

This test ensures the module imports and the main window can be instantiated
without rendering to a display. It can be run as a script or via pytest.
"""
import os
import sys

# Force Qt to use an offscreen platform so no display is required
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.vlm_gui import VLMApp  # noqa: E402


def test_can_instantiate_main_window():
    app = QApplication.instance() or QApplication([])
    window = VLMApp()
    assert window.windowTitle() == "VLM Explorer"
    # Close immediately
    window.close()
    app.quit()


if __name__ == "__main__":
    # Allow running as a script
    test_can_instantiate_main_window()
    print("Smoke test passed.")
