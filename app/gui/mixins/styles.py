from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QLabel
from PySide6.QtCore import Qt


class StyleMixin:
    """Provides the application stylesheet and font setup."""

    def apply_styles(self):
        """Apply futuristic AI-themed styling."""
        # Set application font - try to use a modern system font
        font_families = QFontDatabase.families()
        preferred_fonts = ["Segoe UI", "SF Pro Display", "Roboto", "Arial"]
        
        selected_font = "Segoe UI"  # Default fallback
        for font_name in preferred_fonts:
            if font_name in font_families:
                selected_font = font_name
                break
                
        font = QFont(selected_font, 9)
        self.setFont(font)

        # Futuristic AI-themed styling
        self.setStyleSheet(
            """
            /* Main Window - Deep space gradient with subtle blue accents */
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #0c0e14, stop:0.5 #121624, stop:1 #1a1f35);
                color: #e0e6ed;
            }

            /* Central Widget */
            QWidget {
                background: transparent;
                color: #e0e6ed;
                font-family: 'Segoe UI', 'SF Pro Display', 'Roboto', sans-serif;
            }

            /* Labels */
            QLabel {
                color: #a0c8ff;
                font-weight: 500;
                padding: 2px;
            }

            /* Buttons - Glowing neon effect */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2b4b8f, stop:1 #1a3366);
                border: 1px solid #4671d5;
                border-radius: 6px;
                color: #ffffff;
                font-weight: 600;
                padding: 10px 16px;
                min-height: 20px;
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b6bcf, stop:1 #2b4b8f);
                border: 1px solid #5a8aea;
                color: #ffffff;
            }

            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a3366, stop:1 #0d1b36);
                border: 1px solid #3b6bcf;
            }

            QPushButton:disabled {
                background: #2a2e3a;
                border: 1px solid #3d4356;
                color: #6c7793;
            }

            /* Text Inputs - Holographic effect */
            QTextEdit, QLineEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2433, stop:1 #131824);
                border: 1px solid #2a385a;
                border-radius: 6px;
                color: #e2e8f0;
                padding: 8px;
                selection-background-color: #3b6bcf;
                font-size: 10pt;
            }

            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid #4671d5;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2940, stop:1 #131c2e);
            }

            /* Combo Boxes - Sleek dropdown */
            QComboBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2433, stop:1 #131824);
                border: 1px solid #2a385a;
                border-radius: 6px;
                color: #e2e8f0;
                padding: 8px;
                min-width: 100px;
                font-weight: 500;
            }

            QComboBox:hover {
                border: 1px solid #4671d5;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2940, stop:1 #131c2e);
            }
            
            QComboBox:focus {
                border: 1px solid #4671d5;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2940, stop:1 #131c2e);
            }

            QComboBox::drop-down {
                border: none;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #4671d5;
                margin-right: 5px;
            }

            QComboBox QAbstractItemView {
                background: #1a1f2e;
                border: 1px solid #4671d5;
                border-radius: 4px;
                color: #e2e8f0;
                selection-background-color: #3b6bcf;
                padding: 4px;
            }

            /* Spin Boxes - Digital readout style */
            QSpinBox, QDoubleSpinBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2433, stop:1 #131824);
                border: 1px solid #2a385a;
                border-radius: 6px;
                color: #e2e8f0;
                padding: 6px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-weight: 500;
            }

            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #4671d5;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2940, stop:1 #131c2e);
            }
            
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #4671d5;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2940, stop:1 #131c2e);
            }
            
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background: transparent;
                border: none;
                width: 16px;
                height: 12px;
            }
            
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 4px solid #4671d5;
            }
            
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #4671d5;
            }

            /* Checkboxes - Futuristic toggle */
            QCheckBox {
                color: #a0c8ff;
                spacing: 10px;
                font-weight: 500;
                padding: 5px;
                border-radius: 4px;
            }
            
            QCheckBox:hover {
                background: rgba(26, 31, 46, 0.5);
            }

            QCheckBox::indicator {
                width: 22px;
                height: 22px;
                border: 1px solid #2a385a;
                border-radius: 4px;
                background: rgba(12, 14, 20, 0.7);
            }

            QCheckBox::indicator:hover {
                border: 1px solid #4671d5;
                background: #1e2940;
            }

            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3b6bcf, stop:1 #2b4b8f);
                border: 1px solid #5a8aea;
            }

            QCheckBox::indicator:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4671d5, stop:1 #3b6bcf);
            }

            /* Progress Bar - Pulsing energy effect */
            QProgressBar {
                background: #0c0e14;
                border: 1px solid #2a385a;
                border-radius: 6px;
                text-align: center;
                color: #e2e8f0;
                font-weight: 600;
                min-height: 6px;
                max-height: 6px;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b6bcf, stop:0.5 #4671d5, stop:1 #3b6bcf);
                border-radius: 3px;
                margin: 0px;
            }

            /* Status Bar - Information display */
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0c0e14, stop:1 #131824);
                border-top: 1px solid #2a385a;
                color: #8a9cbe;
                font-size: 9pt;
            }

            QStatusBar QLabel {
                color: #8a9cbe;
                padding: 4px 8px;
                border-right: 1px solid #2a385a;
            }

            /* Image Display Area - Neural network visualization */
            QLabel#imageDisplay {
                background-color: #0c0e14;
                background-image: repeating-linear-gradient(0deg, transparent, transparent 19px, rgba(70, 113, 213, 0.05) 19px, rgba(70, 113, 213, 0.05) 20px),
                                  repeating-linear-gradient(90deg, transparent, transparent 19px, rgba(70, 113, 213, 0.05) 19px, rgba(70, 113, 213, 0.05) 20px);
                border: 1px solid #2a385a;
                border-radius: 8px;
                color: #4671d5;
                font-size: 12pt;
                font-weight: 500;
            }
            
            QLabel#imageDisplay:hover {
                border: 1px solid #4671d5;
            }

            /* Scrollbars - Sleek minimal design */
            QScrollBar:vertical {
                background: #1a1f2e;
                width: 10px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: #2b4b8f;
                border-radius: 5px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: #3b6bcf;
            }

            QScrollBar:horizontal {
                background: #1a1f2e;
                height: 10px;
                border-radius: 5px;
            }

            QScrollBar::handle:horizontal {
                background: #2b4b8f;
                border-radius: 5px;
                min-width: 20px;
            }

            QScrollBar::handle:horizontal:hover {
                background: #3b6bcf;
            }

            /* Message Boxes - Alert system */
            QMessageBox {
                background: #1a1f2e;
                color: #e2e8f0;
            }

            QMessageBox QPushButton {
                min-width: 80px;
                margin: 4px;
            }

            /* Sidebar - Command center */
            QFrame#sidebar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #131824, stop:1 #0c0e14);
                border-right: 1px solid #2a385a;
            }

            /* Sidebar header - System name */
            QLabel#sidebarHeader {
                color: #a0c8ff;
                font-size: 16pt;
                font-weight: bold;
                padding: 20px 0;
                letter-spacing: 2px;
                border-bottom: 1px solid rgba(42, 56, 90, 0.5);
                margin-bottom: 10px;
            }

            /* Navigation Buttons - Control panel */
            QPushButton#navButton {
                background: transparent;
                border: none;
                color: #8a9cbe;
                text-align: left;
                padding: 10px 16px;
                border-radius: 6px;
                font-weight: 500;
            }
            
            QPushButton#navButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e2940, stop:1 #131c2e);
                border: 1px solid #2a385a;
                color: #a0c8ff;
            }
            
            QPushButton#navButton:hover {
                background: rgba(59, 107, 207, 0.15);
                color: #a0c8ff;
            }

            /* Model Hint Label - System suggestions */
            QLabel#modelHint {
                color: #8a9cbe;
                font-size: 9pt;
                padding: 2px 4px;
            }

            /* Assistant message styling for chat interface */
            .assistant-message-container {
                margin: 16px 0;
                text-align: left;
            }
            
            .assistant-message {
                background: linear-gradient(135deg, #1a1f2e 0%, #1e2940 100%);
                color: #e0e6ed;
                padding: 12px 16px;
                border-radius: 12px 12px 12px 2px;
                display: inline-block;
                max-width: 75%;
                word-wrap: break-word;
                border: 1px solid #2a385a;
                position: relative;
            }
            
            .assistant-message::before {
                content: '';
                position: absolute;
                top: -1px;
                left: -1px;
                right: -1px;
                height: 1px;
                background: linear-gradient(90deg, transparent, #4671d5, transparent);
                border-radius: 12px 12px 0 0;
            }
            
            .assistant-timestamp {
                font-size: 10px;
                color: #8a9cbe;
                margin-top: 4px;
                margin-left: 4px;
                display: flex;
                align-items: center;
            }
            
            .assistant-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #4671d5;
                margin-right: 5px;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 0.7; }
                50% { opacity: 1.0; }
                100% { opacity: 0.7; }
            }
            
            /* Page Headers - Section titles */
            QLabel#pageHeader {
                color: #a0c8ff;
                font-size: 18pt;
                font-weight: 700;
                margin: 10px 0 20px 0;
                letter-spacing: 1px;
                padding-bottom: 8px;
                border-bottom: 1px solid rgba(70, 113, 213, 0.5);
                position: relative;
            }
            
            /* Group Headers - Subsection labels */
            QLabel#groupHeader {
                color: #4671d5;
                font-size: 12pt;
                font-weight: bold;
                margin-top: 15px;
                margin-bottom: 5px;
                letter-spacing: 0.5px;
            }
            
            /* Model Status Label - System status */
            QLabel#modelStatus {
                color: #8a9cbe;
                font-size: 10pt;
                padding: 8px;
                background: rgba(26, 31, 46, 0.5);
                border-radius: 6px;
            }
            
            /* Toolbar Label - Section headers in toolbars */
            QLabel#toolbarLabel {
                color: #4671d5;
                font-weight: 600;
                font-size: 10pt;
                letter-spacing: 1px;
            }
            
            /* Primary action button - Main call to action */
            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b6bcf, stop:1 #2b4b8f);
                border: 1px solid #5a8aea;
                border-radius: 6px;
                color: white;
                font-weight: 600;
                padding: 10px 16px;
                letter-spacing: 1px;
                min-height: 20px;
            }
            
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4671d5, stop:1 #3b6bcf);
                border: 1px solid #60a5fa;
            }
            
            QPushButton#primaryButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2b4b8f, stop:1 #1a3366);
                border: 1px solid #3b6bcf;
            }
            
            QPushButton#primaryButton:disabled {
                background: #2a2e3a;
                border: 1px solid #3d4356;
                color: #6c7793;
            }
            
            /* Secondary action button - Less important actions */
            QPushButton#secondaryButton {
                background: rgba(26, 31, 46, 0.7);
                border: 1px solid #2a385a;
                border-radius: 6px;
                color: #a0c8ff;
                font-weight: 500;
                padding: 8px 12px;
                min-height: 18px;
            }
            
            QPushButton#secondaryButton:hover {
                background: rgba(42, 56, 90, 0.8);
                border: 1px solid #3b6bcf;
                color: #ffffff;
            }
            
            QPushButton#secondaryButton:pressed {
                background: rgba(26, 31, 46, 0.9);
                border: 1px solid #2a385a;
            }
            
            /* Action button - Utility actions */
            QPushButton#actionButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2940, stop:1 #131c2e);
                border: 1px solid #2a385a;
                border-radius: 6px;
                color: #a0c8ff;
                font-weight: 500;
                padding: 8px 12px;
                min-height: 18px;
            }
            
            QPushButton#actionButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a385a, stop:1 #1e2940);
                border: 1px solid #3b6bcf;
                color: #ffffff;
            }
            
            QPushButton#actionButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #131c2e, stop:1 #0c0e14);
                border: 1px solid #2a385a;
            }
            
            /* Image path indicator fields */
            QLineEdit[readOnly="true"] {
                background: rgba(12, 14, 20, 0.7);
                border: 1px solid #2a385a;
                border-radius: 4px;
                color: #8a9cbe;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 9pt;
                padding: 6px 8px;
            }
            
            QLineEdit[readOnly="true"]:hover {
                border: 1px solid #3b6bcf;
                background: rgba(26, 31, 46, 0.7);
            }
            
            /* Server Status Frame - Container for server status */
            QFrame#statusFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1f2e, stop:1 #131824);
                border: 1px solid #2a385a;
                border-radius: 8px;
                padding: 10px;
                margin-top: 15px;
                margin-bottom: 15px;
            }
            
            /* Server Status Label - Current server state */
            QLabel#serverStatus {
                color: #a0c8ff;
                font-weight: 600;
                font-size: 11pt;
                padding: 5px;
            }
            
            /* Metrics Frame - Container for metrics display */
            QFrame#metricsFrame {
                background: rgba(12, 14, 20, 0.7);
                border: 1px solid #2a385a;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
            }
            
            /* Metrics Header - Title for metrics section */
            QLabel#metricsHeader {
                color: #4671d5;
                font-weight: 600;
                font-size: 11pt;
                letter-spacing: 1px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(70, 113, 213, 0.3);
                margin-bottom: 10px;
            }
            
            /* Metric Label - Individual metric display */
            QLabel#metricLabel {
                color: #8a9cbe;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 10pt;
                padding: 5px;
                background: rgba(26, 31, 46, 0.5);
                border-radius: 4px;
                margin: 3px;
            }
            
            /* Small Text Edit - For limited height text inputs */
            QTextEdit[maximumHeight="60"] {
                background: rgba(12, 14, 20, 0.7);
                border: 1px solid #2a385a;
                border-radius: 6px;
                color: #a0c8ff;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 9pt;
                padding: 8px;
            }
            
            QTextEdit[maximumHeight="60"]:focus {
                border: 1px solid #4671d5;
                background: rgba(26, 31, 46, 0.7);
            }
            
            /* Form Layouts - Better organization for settings */
            QFormLayout {
                spacing: 12px;
            }
            
            QFormLayout QLabel {
                color: #a0c8ff;
                font-weight: 500;
                min-width: 120px;
            }
            
            /* Settings Section - Group of related controls */
            QWidget > QFormLayout {
                background: rgba(12, 14, 20, 0.3);
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
            }
            """
        )
        
        # Apply drop shadow effects to headers
        self.apply_drop_shadows()

    def apply_drop_shadows(self):
        """Apply drop shadow effects to headers for a glowing effect."""
        try:
            # Find and apply drop shadow to sidebar header
            sidebar_header = self.findChild(QLabel, "sidebarHeader")
            if sidebar_header:
                shadow_effect = QGraphicsDropShadowEffect()
                shadow_effect.setBlurRadius(15)
                shadow_effect.setColor(Qt.GlobalColor.cyan)
                shadow_effect.setOffset(0, 0)
                sidebar_header.setGraphicsEffect(shadow_effect)
            
            # Find and apply drop shadow to page headers
            page_headers = self.findChildren(QLabel, "pageHeader")
            for header in page_headers:
                shadow_effect = QGraphicsDropShadowEffect()
                shadow_effect.setBlurRadius(10)
                shadow_effect.setColor(Qt.GlobalColor.blue)
                shadow_effect.setOffset(0, 0)
                header.setGraphicsEffect(shadow_effect)
                
        except Exception:
            # If drop shadow effects fail, continue without them
            pass
