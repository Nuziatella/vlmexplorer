from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QProgressBar,
)


class ChatPageMixin:
    """Provides the chat/inference UI construction for the main window."""

    def create_chat_page(self) -> QWidget:
        """Create the main chat/inference page with conversational interface."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Top toolbar with model selection and controls
        toolbar = QHBoxLayout()

        # Model selection
        model_label = QLabel("MODEL:")
        model_label.setObjectName("toolbarLabel")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        for model_id, model_name in getattr(self, "available_models", {}).items():
            self.model_combo.addItem(model_name, model_id)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        self.local_model_btn = QPushButton("LOCAL")
        self.local_model_btn.setToolTip("Load a model from a local directory")
        self.local_model_btn.clicked.connect(self.pick_local_model)
        self.local_model_btn.setObjectName("actionButton")

        # Image controls
        self.load_btn = QPushButton("LOAD IMAGE")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setObjectName("actionButton")

        self.screens_combo = QComboBox()
        self.screens_combo.addItems(["One Screen", "Two Screens"])
        self.screens_combo.currentTextChanged.connect(self.on_screens_changed)

        toolbar.addWidget(model_label)
        toolbar.addWidget(self.model_combo)
        toolbar.addWidget(self.local_model_btn)
        toolbar.addWidget(QLabel("|"))
        toolbar.addWidget(self.load_btn)
        toolbar.addWidget(self.screens_combo)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        # Model-specific hint for the chat tab
        self.model_hint_label = QLabel("")
        self.model_hint_label.setObjectName("modelHint")
        layout.addWidget(self.model_hint_label)

        # Local model indicator (hidden by default)
        self.local_model_indicator = QLineEdit()
        self.local_model_indicator.setReadOnly(True)
        self.local_model_indicator.setPlaceholderText("No local model selected")
        self.local_model_indicator.setVisible(False)
        layout.addWidget(self.local_model_indicator)

        # Second image controls (hidden by default)
        self.second_image_layout = QHBoxLayout()
        self.load_btn2 = QPushButton("LOAD SECOND IMAGE")
        self.load_btn2.clicked.connect(self.load_image2)
        self.load_btn2.setObjectName("actionButton")
        self.image2_indicator = QLineEdit()
        self.image2_indicator.setReadOnly(True)
        self.image2_indicator.setPlaceholderText("No second image selected")
        self.second_image_layout.addWidget(self.load_btn2)
        self.second_image_layout.addWidget(self.image2_indicator, 1)

        second_image_widget = QWidget()
        second_image_widget.setLayout(self.second_image_layout)
        second_image_widget.setVisible(False)
        self.second_image_widget = second_image_widget
        layout.addWidget(second_image_widget)

        # Main content area - horizontal split
        main_content = QHBoxLayout()

        # Left side - Image display
        left_panel = QVBoxLayout()

        self.image_label = QLabel(
            "Drop an image here or click Load Image\n\nSupported formats: PNG, JPG, JPEG, BMP, GIF"
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setObjectName("imageDisplay")
        self.image_label.setMinimumHeight(300)
        self.image_label.setMinimumWidth(400)
        self.image_label.setAcceptDrops(False)

        left_panel.addWidget(self.image_label)

        # Right side - Chat interface
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # Chat history area
        chat_label = QLabel("CONVERSATION LOG")
        chat_label.setObjectName("groupHeader")
        right_panel.addWidget(chat_label)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMinimumHeight(250)
        self.chat_history.setPlaceholderText("Chat history will appear here...")
        self.chat_history.setObjectName("chatHistory")
        right_panel.addWidget(self.chat_history)

        # Input area
        input_layout = QHBoxLayout()

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your query about the image...")
        self.prompt_input.returnPressed.connect(self.run_vlm)

        self.run_btn = QPushButton("ANALYZE")
        self.run_btn.clicked.connect(self.run_vlm)
        self.run_btn.setEnabled(False)
        self.run_btn.setMinimumWidth(100)
        self.run_btn.setObjectName("primaryButton")

        input_layout.addWidget(self.prompt_input)
        input_layout.addWidget(self.run_btn)

        right_panel.addLayout(input_layout)

        # Action buttons
        action_layout = QHBoxLayout()

        self.clear_btn = QPushButton("CLEAR")
        self.clear_btn.clicked.connect(self.clear_results)
        self.clear_btn.setToolTip("Clear conversation history")
        self.clear_btn.setObjectName("secondaryButton")

        self.export_btn = QPushButton("EXPORT")
        self.export_btn.setToolTip("Export the latest analysis")
        self.export_btn.clicked.connect(self.export_result)
        self.export_btn.setObjectName("secondaryButton")

        action_layout.addWidget(self.clear_btn)
        action_layout.addWidget(self.export_btn)
        action_layout.addStretch()

        right_panel.addLayout(action_layout)

        # Add panels to main content
        main_content.addLayout(left_panel, 1)
        main_content.addLayout(right_panel, 1)

        layout.addLayout(main_content, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Initialize chat history
        self.conversation_history = []

        # Create placeholder for results display (used by other methods)
        self.results_display = self.chat_history  # Point to chat history

        # Initialize model-aware UI hints on first load
        try:
            self.update_input_placeholder_for_model()
            self.update_model_hint()
        except Exception:
            pass

        return page
