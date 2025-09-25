"""Main entry point for VLM Explorer application.

This launches the enhanced VLM GUI with all features including:
- VRAM monitoring and low-VRAM warnings
- 4-bit and 8-bit quantization support
- FastAPI server integration
- Dual-screen Nintendo DS support
- Pokémon JSON template prompting
"""
import sys
from app.vlm_gui import VLMApp
from PySide6.QtWidgets import QApplication

def main():
    """Launch the VLM Explorer application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = VLMApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
