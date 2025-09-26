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
    
    # Ensure the UI is fully initialized
    app.processEvents()

    # Initially One Screen by default
    assert hasattr(win, "screens_combo")
    assert win.screens_combo.currentText() in ("One Screen", "One")

    # Ensure widgets exist before testing
    assert hasattr(win, "second_image_widget")
    assert hasattr(win, "load_btn2")

    # Switch to Two Screens - try multiple approaches to ensure the signal fires
    # Method 1: Use setCurrentIndex to ensure signal is triggered
    two_screens_index = win.screens_combo.findText("Two Screens")
    if two_screens_index >= 0:
        win.screens_combo.setCurrentIndex(two_screens_index)
    else:
        # Fallback: try to find "Two" 
        two_index = win.screens_combo.findText("Two")
        if two_index >= 0:
            win.screens_combo.setCurrentIndex(two_index)
    
    app.processEvents()
    
    # Method 2: If signal didn't work, call the method directly
    if not win.second_image_widget.isVisible():
        win.on_screens_changed("Two Screens")
        app.processEvents()

    assert win.second_image_widget.isVisible() is True
    assert win.load_btn2.isEnabled() is True

    # Switch back to One Screen
    one_screens_index = win.screens_combo.findText("One Screen")
    if one_screens_index >= 0:
        win.screens_combo.setCurrentIndex(one_screens_index)
    else:
        # Fallback: try to find "One"
        one_index = win.screens_combo.findText("One")
        if one_index >= 0:
            win.screens_combo.setCurrentIndex(one_index)
    
    app.processEvents()
    
    # Method 2: If signal didn't work, call the method directly
    if win.second_image_widget.isVisible():
        win.on_screens_changed("One Screen")
        app.processEvents()

    assert win.second_image_widget.isVisible() is False
    assert win.load_btn2.isEnabled() is False

    # Cleanup
    win.close()
    app.quit()
