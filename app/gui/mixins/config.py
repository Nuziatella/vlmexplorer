from __future__ import annotations

import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QScrollArea,
    QVBoxLayout,
    QLabel,
    QFormLayout,
    QPushButton,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)


class ConfigPageMixin:
    """Provides the advanced configuration page and model-aware UI helpers."""

    def create_config_page(self) -> QScrollArea:
        """Create the config page."""
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        header = QLabel("Advanced Configuration")
        header.setObjectName("pageHeader")
        layout.addWidget(header)

        # Generation settings
        gen_group = QLabel("Generation Settings")
        gen_group.setObjectName("groupHeader")
        layout.addWidget(gen_group)

        gen_form = QFormLayout()

        # Task selection (single source of truth)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["VQA", "Image-to-Text"])
        gen_form.addRow("Task:", self.task_combo)

        self.answer_top_k = QSpinBox()
        self.answer_top_k.setRange(1, 10)
        self.answer_top_k.setValue(1)
        gen_form.addRow("Answer top_k:", self.answer_top_k)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(1.0)
        gen_form.addRow("Temperature:", self.temperature)

        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(0.9)
        gen_form.addRow("Top_p:", self.top_p)

        self.num_beams = QSpinBox()
        self.num_beams.setRange(1, 10)
        self.num_beams.setValue(1)
        gen_form.addRow("Num beams:", self.num_beams)

        self.max_new_tokens = QSpinBox()
        self.max_new_tokens.setRange(1, 512)
        self.max_new_tokens.setValue(32)
        gen_form.addRow("Max new tokens:", self.max_new_tokens)

        layout.addLayout(gen_form)

        # Hardware settings
        hw_group = QLabel("Hardware Settings")
        hw_group.setObjectName("groupHeader")
        layout.addWidget(hw_group)

        self.use_fp16 = QCheckBox("Use FP16 on GPU")
        self.use_fp16.setChecked(torch.cuda.is_available())
        layout.addWidget(self.use_fp16)

        self.use_8bit = QCheckBox("8-bit quantization (bitsandbytes)")
        self.use_8bit.toggled.connect(self.on_toggle_8bit)
        layout.addWidget(self.use_8bit)

        self.use_4bit = QCheckBox("4-bit quantization (bitsandbytes)")
        self.use_4bit.toggled.connect(self.on_toggle_4bit)
        layout.addWidget(self.use_4bit)

        self.device_map_auto = QCheckBox("Accelerate device_map=auto")
        layout.addWidget(self.device_map_auto)

        # Cache settings
        cache_group = QLabel("Cache Settings")
        cache_group.setObjectName("groupHeader")
        layout.addWidget(cache_group)

        cache_form = QFormLayout()
        self.cache_btn = QPushButton("SELECT CACHE")
        self.cache_btn.clicked.connect(self.pick_cache_dir)
        self.cache_btn.setObjectName("actionButton")
        cache_form.addRow("Cache:", self.cache_btn)

        self.cache_indicator = QLineEdit()
        self.cache_indicator.setReadOnly(True)
        self.cache_indicator.setPlaceholderText("Using default HF cache")
        cache_form.addRow("", self.cache_indicator)

        layout.addLayout(cache_form)

        # Image settings
        img_group = QLabel("Image Settings")
        img_group.setObjectName("groupHeader")
        layout.addWidget(img_group)

        img_form = QFormLayout()
        self.max_image_size = QSpinBox()
        self.max_image_size.setRange(0, 4096)
        self.max_image_size.setSingleStep(128)
        self.max_image_size.setValue(0)
        img_form.addRow("Max image size:", self.max_image_size)

        self.cfg_screens_combo = QComboBox()
        self.cfg_screens_combo.addItems(["One", "Two"])
        self.cfg_screens_combo.currentTextChanged.connect(self.on_screens_changed)
        img_form.addRow("Screens:", self.cfg_screens_combo)

        self.cfg_load_btn2 = QPushButton("LOAD IMAGE 2")
        self.cfg_load_btn2.setEnabled(False)
        self.cfg_load_btn2.clicked.connect(self.load_image2)
        self.cfg_load_btn2.setObjectName("actionButton")
        img_form.addRow("", self.cfg_load_btn2)

        self.cfg_image2_indicator = QLineEdit()
        self.cfg_image2_indicator.setReadOnly(True)
        self.cfg_image2_indicator.setPlaceholderText("No second image")
        img_form.addRow("", self.cfg_image2_indicator)

        layout.addLayout(img_form)

        # Advanced prompts
        prompt_group = QLabel("Advanced Prompts")
        prompt_group.setObjectName("groupHeader")
        layout.addWidget(prompt_group)

        layout.addWidget(QLabel("Vision Prompt (optional):"))
        from PySide6.QtWidgets import QTextEdit  # local import to avoid circulars

        self.vision_input = QTextEdit()
        self.vision_input.setPlaceholderText("Optional VISION_PROMPT")
        self.vision_input.setMaximumHeight(60)
        layout.addWidget(self.vision_input)

        layout.addWidget(QLabel("Memory (optional):"))
        self.memory_input = QTextEdit()
        self.memory_input.setPlaceholderText("Optional MEMORY context")
        self.memory_input.setMaximumHeight(60)
        layout.addWidget(self.memory_input)

        layout.addStretch()

        return scroll

    # ---------------- Model-aware UI helpers ----------------
    def update_config_for_model(self) -> None:
        """Update configuration options based on selected model."""
        model_id = getattr(self, "local_model_path", None) or self.model_combo.currentData()
        if not model_id:
            return

        model_name = str(model_id).lower()

        # Update task combo based on model capabilities
        if hasattr(self, "task_combo"):
            self.task_combo.clear()

            if "llava" in model_name or "onevision" in model_name:
                self.task_combo.addItems(["VQA", "Image-to-Text"])
                self.task_combo.setCurrentText("VQA")
            elif "blip" in model_name:
                if "vqa" in model_name:
                    self.task_combo.addItems(["VQA"])
                else:
                    self.task_combo.addItems(["Image-to-Text", "VQA"])
            elif "git" in model_name:
                if "vqa" in model_name:
                    self.task_combo.addItems(["VQA", "Image-to-Text"])
                else:
                    self.task_combo.addItems(["Image-to-Text"])
            elif "vilt" in model_name:
                self.task_combo.addItems(["VQA"])
            else:
                self.task_combo.addItems(["VQA", "Image-to-Text"])

        # Update quantization recommendations
        if hasattr(self, "use_4bit") and hasattr(self, "use_8bit"):
            if any(size in model_name for size in ["7b", "8b", "13b", "large"]):
                if not self.use_4bit.isChecked() and not self.use_8bit.isChecked():
                    self.use_4bit.setChecked(True)

        # Update device mapping for large models
        if hasattr(self, "device_map_auto"):
            if any(size in model_name for size in ["13b", "large"]) and torch.cuda.device_count() > 1:
                self.device_map_auto.setChecked(True)

    def on_toggle_4bit(self, checked: bool) -> None:
        if checked and self.use_8bit.isChecked():
            self.use_8bit.setChecked(False)

    def on_toggle_8bit(self, checked: bool) -> None:
        if checked and self.use_4bit.isChecked():
            self.use_4bit.setChecked(False)

    def pick_cache_dir(self) -> None:
        """Allow the user to select a cache directory for models."""
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Select cache directory")
        if path:
            self.cache_dir = path
            if hasattr(self, "cache_indicator"):
                self.cache_indicator.setText(path)

    def on_screens_changed(self, text: str) -> None:
        """Handle screen count change and refresh hints.

        Note: We maintain separate controls for the chat page (second_image_widget, load_btn2)
        and the config page (cfg_load_btn2, cfg_image2_indicator) to avoid attribute collisions.
        """
        two_mode = text in ("Two", "Two Screens")
        # Toggle chat page second-image UI if present
        if hasattr(self, "second_image_widget"):
            if two_mode:
                self.second_image_widget.show()
            else:
                self.second_image_widget.hide()
            try:
                # Ensure layout recalculates and state propagates
                self.second_image_widget.update()
                self.second_image_widget.repaint()
            except Exception:
                pass
        if hasattr(self, "load_btn2"):
            self.load_btn2.setEnabled(two_mode)
        # Toggle config page second-image loader if present
        if hasattr(self, "cfg_load_btn2"):
            self.cfg_load_btn2.setEnabled(two_mode)
        # Refresh the hint label
        if hasattr(self, "update_model_hint"):
            self.update_model_hint()

    def update_input_placeholder_for_model(self) -> None:
        """Set chat input placeholder based on selected model capabilities."""
        try:
            model_id = getattr(self, "local_model_path", None) or self.model_combo.currentData()
        except Exception:
            model_id = None
        if not model_id:
            if hasattr(self, "prompt_input"):
                self.prompt_input.setPlaceholderText("Ask a question about the image...")
            return

        name = str(model_id).lower()
        placeholder = "Ask a question about the image..."

        if ("llava" in name) or ("onevision" in name) or ("vqa" in name) or ("vilt" in name):
            placeholder = "Ask a visual question, e.g., 'What is Pikachu doing?'"
        elif ("caption" in name) or ("image-caption" in name) or ("vit-gpt2" in name) or ("pix2struct" in name) or (
            "donut" in name
        ) or ("git" in name):
            placeholder = "Describe the image, e.g., 'Generate a concise caption.'"
        elif ("chat" in name) or ("cogvlm" in name) or ("qwen-vl" in name):
            placeholder = "Chat with the image, e.g., 'Summarize the scene.'"

        if hasattr(self, "prompt_input"):
            self.prompt_input.setPlaceholderText(placeholder)

    def update_model_hint(self) -> None:
        """Update a short usage hint on the chat tab based on the selected model."""
        if not hasattr(self, "model_hint_label"):
            return
        try:
            model_id = getattr(self, "local_model_path", None) or self.model_combo.currentData()
        except Exception:
            model_id = None
        task = self.task_combo.currentText() if hasattr(self, "task_combo") else "VQA"
        if not model_id:
            self.model_hint_label.setText("")
            return
        name = str(model_id).lower()

        lines: list[str] = []
        lines.append(f"Model: {model_id}")
        lines.append(f"Task: {task}")

        if ("llava" in name) or ("onevision" in name):
            if task == "VQA":
                lines.append("Hint: Ask a direct question about the image. Example: 'What attack is being used?'")
            else:
                lines.append("Hint: Describe the image concisely. Example: 'Generate a short caption.'")
        elif ("blip" in name) or ("vit-gpt2" in name) or ("git" in name):
            if task == "VQA":
                lines.append("Hint: Ask a specific visual question. Some BLIP variants support VQA.")
            else:
                lines.append("Hint: Provide a caption-style prompt or leave it empty for general description.")
        elif ("vilt" in name):
            lines.append("Hint: VQA-focused. Ask a question about the image.")
        elif ("donut" in name) or ("pix2struct" in name):
            lines.append("Hint: Document understanding. Ask for text/summary from the image.")
        else:
            if task == "VQA":
                lines.append("Hint: Ask a concise question about the image.")
            else:
                lines.append("Hint: Describe the image or ask for a short caption.")

        if hasattr(self, "screens_combo") and self.screens_combo.currentText() in ("Two Screens", "Two"):
            lines.append("Note: Two Screens mode is active. Load both images for best results.")

        hint_text = " \u2022 ".join(lines)
        self.model_hint_label.setText(" • " + hint_text)
