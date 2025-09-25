from PySide6.QtGui import QFont


class StyleMixin:
    """Provides the application stylesheet and font setup."""

    def apply_styles(self):
        """Apply modern dark futuristic blue theme."""
        # Set application font
        font = QFont("Segoe UI", 9)
        self.setFont(font)

        # Modern dark blue futuristic theme
        self.setStyleSheet(
            """
            /* Main Window */
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #0a0a0f, stop:1 #1a1a2e);
                color: #e0e6ed;
            }

            /* Central Widget */
            QWidget {
                background: transparent;
                color: #e0e6ed;
                font-family: 'Segoe UI', sans-serif;
            }

            /* Labels */
            QLabel {
                color: #a8d8ff;
                font-weight: 500;
                padding: 2px;
            }

            /* Buttons */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2563eb, stop:1 #1d4ed8);
                border: 2px solid #3b82f6;
                border-radius: 8px;
                color: white;
                font-weight: 600;
                padding: 10px 16px;
                min-height: 20px;
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border: 2px solid #60a5fa;
            }

            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1d4ed8, stop:1 #1e40af);
                border: 2px solid #2563eb;
            }

            QPushButton:disabled {
                background: #374151;
                border: 2px solid #4b5563;
                color: #9ca3af;
            }

            /* Text Inputs */
            QTextEdit, QLineEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border: 2px solid #334155;
                border-radius: 8px;
                color: #e2e8f0;
                padding: 8px;
                selection-background-color: #3b82f6;
                font-size: 10pt;
            }

            QTextEdit:focus, QLineEdit:focus {
                border: 2px solid #3b82f6;
            }

            /* Combo Boxes */
            QComboBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border: 2px solid #334155;
                border-radius: 8px;
                color: #e2e8f0;
                padding: 8px;
                min-width: 100px;
            }

            QComboBox:hover {
                border: 2px solid #3b82f6;
            }

            QComboBox::drop-down {
                border: none;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #3b82f6;
                margin-right: 5px;
            }

            QComboBox QAbstractItemView {
                background: #1e293b;
                border: 2px solid #3b82f6;
                border-radius: 8px;
                color: #e2e8f0;
                selection-background-color: #3b82f6;
            }

            /* Spin Boxes */
            QSpinBox, QDoubleSpinBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border: 2px solid #334155;
                border-radius: 8px;
                color: #e2e8f0;
                padding: 6px;
            }

            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #3b82f6;
            }

            /* Checkboxes */
            QCheckBox {
                color: #a8d8ff;
                spacing: 8px;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #334155;
                border-radius: 4px;
                background: #1e293b;
            }

            QCheckBox::indicator:hover {
                border: 2px solid #3b82f6;
                background: #1e40af;
            }

            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3b82f6, stop:1 #1d4ed8);
                border: 2px solid #60a5fa;
            }

            QCheckBox::indicator:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #60a5fa, stop:1 #3b82f6);
            }

            /* Progress Bar */
            QProgressBar {
                background: #1e293b;
                border: 2px solid #334155;
                border-radius: 8px;
                text-align: center;
                color: #e2e8f0;
                font-weight: 600;
                min-height: 20px;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #06b6d4);
                border-radius: 6px;
                margin: 2px;
            }

            /* Status Bar */
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f172a, stop:1 #1e293b);
                border-top: 2px solid #334155;
                color: #94a3b8;
                font-size: 9pt;
            }

            QStatusBar QLabel {
                color: #94a3b8;
                padding: 4px 8px;
                border-right: 1px solid #334155;
            }

            /* Image Display Area */
            QLabel#imageDisplay {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f172a, stop:1 #1e293b);
                border: 3px dashed #3b82f6;
                border-radius: 12px;
                color: #60a5fa;
                font-size: 12pt;
                font-weight: 500;
            }

            /* Scrollbars */
            QScrollBar:vertical {
                background: #1e293b;
                width: 12px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical {
                background: #3b82f6;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: #60a5fa;
            }

            QScrollBar:horizontal {
                background: #1e293b;
                height: 12px;
                border-radius: 6px;
            }

            QScrollBar::handle:horizontal {
                background: #3b82f6;
                border-radius: 6px;
                min-width: 20px;
            }

            QScrollBar::handle:horizontal:hover {
                background: #60a5fa;
            }

            /* Message Boxes */
            QMessageBox {
                background: #1e293b;
                color: #e2e8f0;
            }

            QMessageBox QPushButton {
                min-width: 80px;
                margin: 4px;
            }

            /* Sidebar */
            QFrame#sidebar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border-right: 2px solid #334155;
            }

            /* Sidebar header */
            QLabel#sidebarHeader {
                color: #93c5fd;
                font-size: 16pt;
                font-weight: bold;
                padding: 16px 0;
            }

            /* Navigation Buttons */
            QPushButton#navButton {
                background: transparent;
                border: none;
                color: #cbd5e1;
                text-align: left;
                padding: 10px 16px;
                border-radius: 6px;
            }

            QPushButton#navButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e293b, stop:1 #0f172a);
                border: 1px solid #334155;
                color: #e2e8f0;
            }

            QPushButton#navButton:hover {
                background: rgba(59, 130, 246, 0.15);
                color: #e2e8f0;
            }

            /* Page Headers */
            QLabel#pageHeader {
                color: #93c5fd;
                font-size: 18pt;
                font-weight: 700;
                margin: 10px 0 20px 0;
            }

            /* Group Headers */
            QLabel#groupHeader {
                color: #60a5fa;
                font-size: 12pt;
                font-weight: bold;
                margin-top: 15px;
                margin-bottom: 5px;
            }

            /* Chat Interface */
            QTextEdit#chatHistory {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f172a, stop:1 #1e293b);
                border: 2px solid #334155;
                border-radius: 12px;
                padding: 10px;
                color: #e2e8f0;
                font-size: 10pt;
                selection-background-color: #3b82f6;
            }

            /* Model Hint Label */
            QLabel#modelHint {
                color: #94a3b8;
                font-size: 9pt;
                padding: 2px 4px;
            }

            QLineEdit {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e293b, stop:1 #334155);
                border: 2px solid #475569;
                border-radius: 8px;
                padding: 8px 12px;
                color: #e2e8f0;
                font-size: 10pt;
                min-height: 16px;
            }

            QLineEdit:focus {
                border: 2px solid #3b82f6;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #334155, stop:1 #475569);
            }
            """
        )
