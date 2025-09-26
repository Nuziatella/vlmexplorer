import json
from PySide6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QWidget,
    QMessageBox,
    QStackedWidget,
    QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from dotenv import load_dotenv
from app.gui.models_catalog import AVAILABLE_MODELS
from app.gui.mixins.styles import StyleMixin
from app.gui.mixins.server_mixin import ServerPageMixin
from app.gui.mixins.statusbar import StatusBarMixin
from app.gui.mixins.models import ModelsPageMixin
from app.gui.mixins.chat import ChatPageMixin
from app.gui.mixins.config import ConfigPageMixin
from app.gui.mixins.media import MediaMixin
from app.gui.mixins.inference import InferenceMixin

# Load environment variables
load_dotenv()

class VLMApp(
    ServerPageMixin,
    StyleMixin,
    StatusBarMixin,
    ModelsPageMixin,
    ChatPageMixin,
    ConfigPageMixin,
    MediaMixin,
    InferenceMixin,
    QMainWindow,
):
    """
    Main application window for interacting with Hugging Face VLMs.

    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLM Explorer")
        self.setMinimumSize(1200, 900)
        
        # Set application icon
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icon_path = os.path.join(project_root, "favicon_io", "favicon-32x32.png")
        try:
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass  # Fallback if icon not found
        
        # Available VLM models (centralized catalog)
        self.available_models = AVAILABLE_MODELS
        
        self.current_image_path = None
        self.worker = None
        self.local_model_path = None
        self.cache_dir = None
        self.server_process = None
        self.server_thread = None
        self.server_port = 8001  # Use 8001 instead of 8000 to avoid Windows port conflicts
        
        # Model loading state
        self.loaded_pipeline = None
        self.loaded_model_name = None
        self.model_load_worker = None
        self.loaded_model_config = None  # Store config used when loading model
        
        self.init_ui()
        self.init_statusbar()
        self.init_vram_timer()
    
    def init_ui(self):
        """Build and arrange all widgets with sidebar navigation."""
        # Main container with horizontal layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create sidebar
        self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        # Create main content area
        self.content_stack = QStackedWidget()
        self.create_content_pages()
        main_layout.addWidget(self.content_stack, 1)
        
        self.setCentralWidget(main_widget)
        self.apply_styles()
        self.setAcceptDrops(True)
        self.last_answer = None
        self.show_chat_page()

    def create_sidebar(self):
        """Create the left sidebar navigation."""
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(250)
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("VLM EXPLORER")
        header.setObjectName("sidebarHeader")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(header)
        
        # Navigation buttons
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(10, 20, 10, 10)
        nav_layout.setSpacing(5)
        
        self.nav_buttons = {}
        nav_items = [
            ("Chat", self.show_chat_page),
            ("Models", self.show_models_page),
            ("Configuration", self.show_config_page),
            ("Server", self.show_server_page)
        ]
        
        for name, callback in nav_items:
            btn = QPushButton(name)
            btn.setObjectName("navButton")
            btn.clicked.connect(callback)
            btn.setCheckable(True)
            nav_layout.addWidget(btn)
            self.nav_buttons[name] = btn
        
        nav_layout.addStretch()
        sidebar_layout.addLayout(nav_layout)

    def create_content_pages(self):
        """Create the main content pages."""
        self.chat_page = ChatPageMixin.create_chat_page(self)
        self.content_stack.addWidget(self.chat_page)
        
        self.models_page = self.create_models_page()
        self.content_stack.addWidget(self.models_page)
        
        self.config_page = self.create_config_page()
        self.content_stack.addWidget(self.config_page)
        
        self.server_page = self.create_server_page()
        self.content_stack.addWidget(self.server_page)

    

    def on_model_changed(self):
        """Handle model selection change."""
        # Update config tab based on selected model
        self.update_config_for_model()
        # Update chat input placeholder to match selected model behavior
        self.update_input_placeholder_for_model()
        # Update model-specific hint on the chat tab
        self.update_model_hint()
        
        # Clear chat history when model changes
        if hasattr(self, 'conversation_history'):
            self.conversation_history.clear()
            if hasattr(self, 'chat_history'):
                self.chat_history.clear()
    
    def update_config_for_model(self):
        """Delegate to ConfigPageMixin for model-aware config updates."""
        return ConfigPageMixin.update_config_for_model(self)

    def update_input_placeholder_for_model(self):
        """Delegate to ConfigPageMixin for placeholder updates."""
        return ConfigPageMixin.update_input_placeholder_for_model(self)

    def update_model_hint(self):
        """Delegate to ConfigPageMixin for model hint updates."""
        return ConfigPageMixin.update_model_hint(self)

    def create_models_page(self):
        """Delegate to ModelsPageMixin to build the models page."""
        return ModelsPageMixin.create_models_page(self)

    def create_config_page(self):
        """Delegate to ConfigPageMixin to build the config page."""
        return ConfigPageMixin.create_config_page(self)

    def create_server_page(self):
        """Delegate to ServerPageMixin to build the server page."""
        return ServerPageMixin.create_server_page(self)

    def show_chat_page(self):
        """Show chat page."""
        self.content_stack.setCurrentWidget(self.chat_page)
        self.update_nav_buttons("Chat")

    def show_models_page(self):
        """Show models page."""
        self.content_stack.setCurrentWidget(self.models_page)
        self.update_nav_buttons("Models")

    def show_config_page(self):
        """Show config page."""
        self.content_stack.setCurrentWidget(self.config_page)
        self.update_nav_buttons("Configuration")

    def show_server_page(self):
        """Show server page."""
        self.content_stack.setCurrentWidget(self.server_page)
        self.update_nav_buttons("Server")

    def update_nav_buttons(self, active_name):
        """Update navigation button states."""
        for name, btn in self.nav_buttons.items():
            btn.setChecked(name == active_name)

    def load_model(self):
        """Delegate to ModelsPageMixin."""
        return ModelsPageMixin.load_model(self)
    
    def on_model_loaded(self, pipeline_obj):
        """Delegate to ModelsPageMixin."""
        return ModelsPageMixin.on_model_loaded(self, pipeline_obj)
    
    def on_model_load_error(self, error_msg):
        """Delegate to ModelsPageMixin."""
        return ModelsPageMixin.on_model_load_error(self, error_msg)
    
    def on_model_load_progress(self, value):
        """Delegate to ModelsPageMixin."""
        return ModelsPageMixin.on_model_load_progress(self, value)
    
    def unload_model(self):
        """Delegate to ModelsPageMixin."""
        return ModelsPageMixin.unload_model(self)

    def config_matches_loaded_model(self):
        """Delegate to ModelsPageMixin."""
        return ModelsPageMixin.config_matches_loaded_model(self)

    def browse_hf_models(self):
        """Populate the model combo with top community models for the selected task."""
        try:
            from huggingface_hub import list_models

            task = "visual-question-answering" if self.task_combo.currentText() == "VQA" else "image-to-text"
            models = list_models(filter=task, sort="downloads", direction=-1, limit=15)
            self.model_combo.clear()
            for m in models:
                name = getattr(m, "modelId", str(m))
                self.model_combo.addItem(name, name)
            QMessageBox.information(self, "Hugging Face", f"Loaded {self.model_combo.count()} models for {task}.")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Hugging Face", f"Failed to list models: {e}")

    def pick_local_model(self):
        """Delegate to MediaMixin."""
        return MediaMixin.pick_local_model(self)

    def init_statusbar(self) -> None:
        """Delegate to StatusBarMixin to set up status bar widgets."""
        return StatusBarMixin.init_statusbar(self)

    def init_vram_timer(self) -> None:
        """Delegate to StatusBarMixin to start VRAM polling timer."""
        return StatusBarMixin.init_vram_timer(self)

    def update_vram_status(self) -> None:
        """Delegate to StatusBarMixin to refresh VRAM label."""
        return StatusBarMixin.update_vram_status(self)

    def export_result(self):
        """Export the conversation (or latest answer) to a JSON file."""
        try:
            model_id = getattr(self, 'local_model_path', None) or self.model_combo.currentData()
        except Exception:
            model_id = None

        data = {
            "model": model_id,
            "screens": self.screens_combo.currentText() if hasattr(self, 'screens_combo') else None,
            "image": getattr(self, 'current_image_path', None),
            "image2": getattr(self, 'image2_path', None),
            "gen_params": {
                "answer_top_k": self.answer_top_k.value() if hasattr(self, 'answer_top_k') else None,
                "temperature": self.temperature.value() if hasattr(self, 'temperature') else None,
                "top_p": self.top_p.value() if hasattr(self, 'top_p') else None,
                "num_beams": self.num_beams.value() if hasattr(self, 'num_beams') else None,
                "max_new_tokens": self.max_new_tokens.value() if hasattr(self, 'max_new_tokens') else None,
            },
            "messages": list(self.conversation_history) if hasattr(self, 'conversation_history') else [],
            "last_answer": getattr(self, 'last_answer', None),
        }

        if not data["messages"] and not data["last_answer"]:
            QMessageBox.information(self, "Export", "No result to export yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save result", "result.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Export", f"Saved to {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export", f"Failed to save: {e}")

    def apply_styles(self):
        """Delegate to StyleMixin for styles."""
        return StyleMixin.apply_styles(self)
    
    def load_image(self):
        return MediaMixin.load_image(self)
    
    def load_image2(self):
        """Delegate to MediaMixin."""
        return MediaMixin.load_image2(self)
    
    def dragEnterEvent(self, event):
        """Delegate to MediaMixin."""
        return MediaMixin.dragEnterEvent(self, event)

    def dropEvent(self, event):
        """Delegate to MediaMixin."""
        return MediaMixin.dropEvent(self, event)

    def resizeEvent(self, event):
        """Delegate to MediaMixin."""
        return MediaMixin.resizeEvent(self, event)

    def run_vlm(self):
        """Delegate to InferenceMixin."""
        return InferenceMixin.run_vlm(self)
    
    def update_progress(self, value):
        """Delegate to InferenceMixin."""
        return InferenceMixin.update_progress(self, value)
    
    def get_input_text(self, widget) -> str:
        """Delegate to InferenceMixin."""
        return InferenceMixin.get_input_text(self, widget)
    
    def on_vlm_finished(self, result):
        """Delegate to InferenceMixin."""
        return InferenceMixin.on_vlm_finished(self, result)
    
    def get_timestamp(self):
        """Delegate to InferenceMixin."""
        return InferenceMixin.get_timestamp(self)
    
    def update_chat_display(self):
        """Delegate to InferenceMixin."""
        return InferenceMixin.update_chat_display(self)

    def on_vlm_timing(self, seconds: float) -> None:
        return InferenceMixin.on_vlm_timing(self, seconds)
    
    def on_vlm_error(self, error_msg):
        """Delegate to InferenceMixin."""
        return InferenceMixin.on_vlm_error(self, error_msg)
    
    def cleanup_worker(self):
        """Delegate to InferenceMixin."""
        return InferenceMixin.cleanup_worker(self)
    
    def set_ui_enabled(self, enabled):
        """Delegate to InferenceMixin."""
        return InferenceMixin.set_ui_enabled(self, enabled)
    
    def closeEvent(self, event):
        """Ensure the worker thread and server are terminated cleanly on close."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.stop_server()
        event.accept()

    def clear_results(self):
        """Clear the conversation/results view."""
        if hasattr(self, 'conversation_history'):
            self.conversation_history.clear()
        if hasattr(self, 'chat_history'):
            self.chat_history.clear()
        if hasattr(self, 'results_display'):
            self.results_display.clear()
        self.last_answer = None

    def on_screens_changed(self, text: str):
        """Delegate to ConfigPageMixin for screen count changes."""
        return ConfigPageMixin.on_screens_changed(self, text)

    def build_prompt(self, vision_text: str, memory_text: str, user_question: str) -> str:
        """Delegate to InferenceMixin."""
        return InferenceMixin.build_prompt(self, vision_text, memory_text, user_question)

    def toggle_server(self):
        """Delegate to ServerPageMixin."""
        return ServerPageMixin.toggle_server(self)

    def start_server(self):
        """Delegate to ServerPageMixin."""
        return ServerPageMixin.start_server(self)

    def stop_server(self):
        """Delegate to ServerPageMixin."""
        return ServerPageMixin.stop_server(self)

    
