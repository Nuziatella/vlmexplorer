from __future__ import annotations

from typing import Any

import torch
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFormLayout,
    QComboBox,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QMessageBox,
)

from app.vlm_worker import ModelLoader


class ModelsPageMixin:
    """Provides the Models page UI and model load/unload controls."""

    # ---------- UI construction ----------
    def create_models_page(self) -> QWidget:
        """Create the models page with model selection, cache display, and load/unload controls."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        header = QLabel("Model Configuration")
        header.setObjectName("pageHeader")
        layout.addWidget(header)

        # Model selection form
        form_layout = QFormLayout()

        # Model dropdown (already created in chat page, but keep it here too if missing)
        if not hasattr(self, "model_combo"):
            self.model_combo = QComboBox()
            for model_id, model_name in getattr(self, "available_models", {}).items():
                self.model_combo.addItem(model_name, model_id)
        form_layout.addRow("Model:", self.model_combo)

        # Local model selection button
        if not hasattr(self, "local_model_btn"):
            self.local_model_btn = QPushButton("Browse Local Model...")
            self.local_model_btn.clicked.connect(self.pick_local_model)
        form_layout.addRow("Local Model:", self.local_model_btn)

        self.local_model_indicator = QLineEdit()
        self.local_model_indicator.setReadOnly(True)
        self.local_model_indicator.setPlaceholderText("No local model selected")
        form_layout.addRow("", self.local_model_indicator)

        # Browse community models from Hugging Face
        self.browse_models_btn = QPushButton("🔍 Browse HF Models")
        self.browse_models_btn.setToolTip("Fetch popular models for the selected task")
        self.browse_models_btn.clicked.connect(self.browse_hf_models)
        form_layout.addRow("", self.browse_models_btn)

        # Task selection is configured on the Configuration page to avoid duplicate widgets.

        layout.addLayout(form_layout)

        # Model loading section
        load_group = QLabel("Model Loading")
        load_group.setObjectName("groupHeader")
        layout.addWidget(load_group)

        load_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("🔄 Load Model")
        self.load_model_btn.setToolTip("Pre-load the selected model for faster inference")
        self.load_model_btn.clicked.connect(self.load_model)

        self.unload_model_btn = QPushButton("❌ Unload Model")
        self.unload_model_btn.setToolTip("Unload the current model to free memory")
        self.unload_model_btn.clicked.connect(self.unload_model)
        self.unload_model_btn.setEnabled(False)

        load_layout.addWidget(self.load_model_btn)
        load_layout.addWidget(self.unload_model_btn)
        load_layout.addStretch()
        layout.addLayout(load_layout)

        # Model status
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setObjectName("modelStatus")
        layout.addWidget(self.model_status_label)

        layout.addStretch()
        return page

    # ---------- Loading logic ----------
    def load_model(self) -> None:
        """Load the selected model in the background."""
        model_name = getattr(self, "local_model_path", None) or self.model_combo.currentData()
        if not model_name:
            QMessageBox.warning(self, "No Model Selected", "Please select a model first.")
            return

        if getattr(self, "model_load_worker", None) and self.model_load_worker.isRunning():
            QMessageBox.information(self, "Loading in Progress", "A model is already being loaded.")
            return

        # Collect load options
        load_opts: dict[str, Any] = {
            "use_8bit": self.use_8bit.isChecked(),
            "use_4bit": self.use_4bit.isChecked(),
            "device_map_auto": self.device_map_auto.isChecked(),
        }

        # Start loading
        self.model_load_worker = ModelLoader(model_name, getattr(self, "cache_dir", None), load_opts)
        self.model_load_worker.finished.connect(self.on_model_loaded)
        self.model_load_worker.error.connect(self.on_model_load_error)
        self.model_load_worker.progress.connect(self.on_model_load_progress)

        # Update UI
        self.load_model_btn.setEnabled(False)
        self.model_status_label.setText("Loading model...")
        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

        self.model_load_worker.start()

    def on_model_loaded(self, pipeline_obj) -> None:
        """Handle successful model loading."""
        self.loaded_pipeline = pipeline_obj
        self.loaded_model_name = getattr(self, "local_model_path", None) or self.model_combo.currentData()

        # Store the configuration used for loading
        self.loaded_model_config = {
            "use_8bit": self.use_8bit.isChecked(),
            "use_4bit": self.use_4bit.isChecked(),
            "device_map_auto": self.device_map_auto.isChecked(),
            "use_fp16": self.use_fp16.isChecked(),
        }

        # Update UI
        self.load_model_btn.setEnabled(True)
        self.unload_model_btn.setEnabled(True)
        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(False)

        model_display_name = self.loaded_model_name
        if isinstance(model_display_name, str) and len(model_display_name) > 50:
            model_display_name = "..." + model_display_name[-47:]

        self.model_status_label.setText(f"✅ Loaded: {model_display_name}")

        QMessageBox.information(
            self,
            "Model Loaded",
            f"Model '{self.loaded_model_name}' loaded successfully!\n"
            "Inference will now be much faster.",
        )

    def on_model_load_error(self, error_msg: str) -> None:
        """Handle model loading error."""
        self.load_model_btn.setEnabled(True)
        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(False)
        self.model_status_label.setText("❌ Load failed")

        QMessageBox.critical(self, "Model Load Error", f"Failed to load model:\n{error_msg}")

    def on_model_load_progress(self, value: int) -> None:
        """Update model loading progress."""
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(value)

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if getattr(self, "loaded_pipeline", None):
            # Clear the pipeline and force garbage collection
            self.loaded_pipeline = None
            self.loaded_model_name = None

            # Force cleanup
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update UI
            self.unload_model_btn.setEnabled(False)
            self.model_status_label.setText("No model loaded")

            QMessageBox.information(self, "Model Unloaded", "Model unloaded and memory freed.")

    def config_matches_loaded_model(self) -> bool:
        """Check if current configuration matches the loaded model's configuration."""
        if not getattr(self, "loaded_model_config", None):
            return False

        current_config = {
            "use_8bit": self.use_8bit.isChecked(),
            "use_4bit": self.use_4bit.isChecked(),
            "device_map_auto": self.device_map_auto.isChecked(),
            "use_fp16": self.use_fp16.isChecked(),
        }

        return current_config == self.loaded_model_config
