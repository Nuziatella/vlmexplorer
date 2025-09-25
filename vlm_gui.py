import sys
import subprocess
import threading
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QTextEdit, QWidget,
                               QComboBox, QProgressBar, QMessageBox, QSpinBox,
                               QDoubleSpinBox, QFormLayout, QLineEdit, QCheckBox,
                               QStackedWidget, QScrollArea, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon, QFont
from PySide6.QtCore import QTimer
from PIL import Image
import torch
from app.vlm_worker import VLMWorker, ModelLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VLMApp(QMainWindow):
    """
    Main application window for interacting with Hugging Face VLMs.

    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLM Explorer - Vision Language Model Interface")
        self.setMinimumSize(1200, 900)
        
        # Set application icon
        icon_path = "D:/CascadeProjects/windsurf-project/favicon_io/favicon-32x32.png"
        try:
            self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass  # Fallback if icon not found
        
        # Available VLM models (comprehensive list of supported models)
        self.available_models = {
            # BLIP Models (Image Captioning & VQA)
            "Salesforce/blip-image-captioning-base": "BLIP Base - Image Captioning",
            "Salesforce/blip-image-captioning-large": "BLIP Large - Image Captioning",
            "Salesforce/blip-vqa-base": "BLIP Base - Visual Question Answering",
            "Salesforce/blip-vqa-capfilt-large": "BLIP Large - VQA (Caption Filtered)",
            
            # BLIP-2 Models (Advanced VQA & Captioning)
            "Salesforce/blip2-opt-2.7b": "BLIP-2 OPT 2.7B - Conversational AI",
            "Salesforce/blip2-opt-6.7b": "BLIP-2 OPT 6.7B - Advanced Conversational",
            "Salesforce/blip2-flan-t5-base": "BLIP-2 FLAN-T5 Base - Instruction Following",
            "Salesforce/blip2-flan-t5-xl": "BLIP-2 FLAN-T5 XL - Large Instruction Model",
            
            # InstructBLIP (Instruction-tuned)
            "Salesforce/instructblip-vicuna-7b": "InstructBLIP Vicuna 7B - Instruction Following",
            "Salesforce/instructblip-vicuna-13b": "InstructBLIP Vicuna 13B - Large Instruction",
            "Salesforce/instructblip-flan-t5-base": "InstructBLIP FLAN-T5 Base",
            "Salesforce/instructblip-flan-t5-xl": "InstructBLIP FLAN-T5 XL",
            
            # ViLT (Vision-and-Language Transformer)
            "dandelin/vilt-b32-finetuned-vqa": "ViLT Base - VQA Finetuned",
            "dandelin/vilt-b32-finetuned-nlvr2": "ViLT Base - NLVR2 Reasoning",
            
            # GIT (GenerativeImage2Text)
            "microsoft/git-base": "GIT Base - Image to Text Generation",
            "microsoft/git-base-coco": "GIT Base - COCO Trained",
            "microsoft/git-base-textcaps": "GIT Base - TextCaps (OCR)",
            "microsoft/git-base-vqav2": "GIT Base - VQA v2",
            "microsoft/git-large": "GIT Large - Advanced Generation",
            "microsoft/git-large-coco": "GIT Large - COCO Trained",
            "microsoft/git-large-textcaps": "GIT Large - TextCaps (OCR)",
            "microsoft/git-large-vqav2": "GIT Large - VQA v2",
            
            # ViT-GPT2 (Vision Transformer + GPT-2)
            "nlpconnect/vit-gpt2-image-captioning": "ViT-GPT2 - Image Captioning",
            
            # CLIP Interrogator (Image to Prompt)
            "pharma/CLIP-Interrogator": "CLIP Interrogator - Image to Prompt",
            
            # Flamingo-style Models
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b": "OpenFlamingo 3B - Few-shot Learning",
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b": "OpenFlamingo 4B - Advanced Few-shot",
            "openflamingo/OpenFlamingo-9B-vitl-mpt7b": "OpenFlamingo 9B - Large Few-shot",
            
            # LLaVA Models (Conversation & VQA)
            "llava-hf/llava-1.5-7b-hf": "LLaVA 1.5 7B - Conversational VLM",
            "llava-hf/llava-1.5-13b-hf": "LLaVA 1.5 13B - Large Conversational",
            "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA 1.6 Mistral 7B - Latest",
            "llava-hf/llava-v1.6-vicuna-7b-hf": "LLaVA 1.6 Vicuna 7B - Improved",
            "llava-hf/llava-v1.6-vicuna-13b-hf": "LLaVA 1.6 Vicuna 13B - Large Improved",
            
            # Fuyu Models (Multimodal)
            "adept/fuyu-8b": "Fuyu 8B - Multimodal Understanding",
            
            # CogVLM Models
            "THUDM/cogvlm-chat-hf": "CogVLM Chat - Conversational Vision",
            
            # Qwen-VL Models
            "Qwen/Qwen-VL": "Qwen-VL - Alibaba Vision Language",
            "Qwen/Qwen-VL-Chat": "Qwen-VL Chat - Conversational",
            
            # MiniGPT Models
            "Vision-CAIR/MiniGPT-4": "MiniGPT-4 - Conversational Vision",
            
            # KOSMOS Models
            "microsoft/kosmos-2-patch14-224": "KOSMOS-2 - Multimodal Large Language Model",
            
            # Pix2Struct Models (Document Understanding)
            "google/pix2struct-base": "Pix2Struct Base - Document AI",
            "google/pix2struct-large": "Pix2Struct Large - Advanced Document AI",
            "google/pix2struct-textcaps-base": "Pix2Struct TextCaps - OCR Focused",
            "google/pix2struct-ai2d-base": "Pix2Struct AI2D - Diagram Understanding",
            "google/pix2struct-docvqa-base": "Pix2Struct DocVQA - Document QA",
            "google/pix2struct-infographicvqa-base": "Pix2Struct InfoVQA - Infographic QA",
            
            # DonutSwin Models (Document Understanding)
            "naver-clova-ix/donut-base": "Donut Base - Document Understanding",
            "naver-clova-ix/donut-base-finetuned-docvqa": "Donut DocVQA - Document QA",
            "naver-clova-ix/donut-base-finetuned-cord-v2": "Donut CORD - Receipt Understanding",
            
            # TrOCR Models (OCR)
            "microsoft/trocr-base-handwritten": "TrOCR Base - Handwritten Text",
            "microsoft/trocr-large-handwritten": "TrOCR Large - Advanced Handwritten",
            "microsoft/trocr-base-printed": "TrOCR Base - Printed Text",
            "microsoft/trocr-large-printed": "TrOCR Large - Advanced Printed",
            
            # LayoutLM Models (Document Layout)
            "microsoft/layoutlmv3-base": "LayoutLMv3 Base - Document Layout",
            "microsoft/layoutlmv3-large": "LayoutLMv3 Large - Advanced Layout",
        }
        
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
        header = QLabel("VLM Explorer")
        header.setObjectName("sidebarHeader")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(header)
        
        # Navigation buttons
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(10, 20, 10, 10)
        nav_layout.setSpacing(5)
        
        self.nav_buttons = {}
        nav_items = [
            ("💬", "Chat", self.show_chat_page),
            ("🤖", "Models", self.show_models_page),
            ("⚙️", "Configuration", self.show_config_page),
            ("🚀", "Server", self.show_server_page)
        ]
        
        for icon, name, callback in nav_items:
            btn = QPushButton(f"{icon}  {name}")
            btn.setObjectName("navButton")
            btn.clicked.connect(callback)
            btn.setCheckable(True)
            nav_layout.addWidget(btn)
            self.nav_buttons[name] = btn
        
        nav_layout.addStretch()
        sidebar_layout.addLayout(nav_layout)

    def create_content_pages(self):
        """Create the main content pages."""
        self.chat_page = self.create_chat_page()
        self.content_stack.addWidget(self.chat_page)
        
        self.models_page = self.create_models_page()
        self.content_stack.addWidget(self.models_page)
        
        self.config_page = self.create_config_page()
        self.content_stack.addWidget(self.config_page)
        
        self.server_page = self.create_server_page()
        self.content_stack.addWidget(self.server_page)

    def create_chat_page(self):
        """Create the main chat/inference page with conversational interface."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Top toolbar with model selection and controls
        toolbar = QHBoxLayout()
        
        # Model selection
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        for model_id, model_name in self.available_models.items():
            self.model_combo.addItem(model_name, model_id)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        
        self.local_model_btn = QPushButton("📁 Local")
        self.local_model_btn.setToolTip("Load a model from a local directory")
        self.local_model_btn.clicked.connect(self.pick_local_model)
        
        # Image controls
        self.load_btn = QPushButton("🖼️ Load Image")
        self.load_btn.clicked.connect(self.load_image)
        
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
        self.load_btn2 = QPushButton("📁 Load Second Image")
        self.load_btn2.clicked.connect(self.load_image2)
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
        
        self.image_label = QLabel("🖼️ Drop an image here or click Load Image\n\nSupported formats: PNG, JPG, JPEG, BMP, GIF")
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
        chat_label = QLabel("💬 Conversation")
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
        self.prompt_input.setPlaceholderText("Ask a question about the image...")
        self.prompt_input.returnPressed.connect(self.run_vlm)
        
        self.run_btn = QPushButton("Send")
        self.run_btn.clicked.connect(self.run_vlm)
        self.run_btn.setEnabled(False)
        self.run_btn.setMinimumWidth(80)
        
        input_layout.addWidget(self.prompt_input)
        input_layout.addWidget(self.run_btn)
        
        right_panel.addLayout(input_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑️ Clear Chat")
        self.clear_btn.clicked.connect(self.clear_results)
        
        self.export_btn = QPushButton("💾 Export")
        self.export_btn.setToolTip("Export the latest response")
        self.export_btn.clicked.connect(self.export_result)
        
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
        
        # Initialize chat history and create placeholders for config elements
        self.conversation_history = []
        
        # Create config elements that are needed by other parts of the code
        self.answer_top_k = QSpinBox()
        self.answer_top_k.setRange(1, 10)
        self.answer_top_k.setValue(1)
        self.use_fp16 = QCheckBox("Use FP16 on GPU")
        self.use_fp16.setChecked(torch.cuda.is_available())
        self.use_8bit = QCheckBox("8-bit quantization")
        self.use_4bit = QCheckBox("4-bit quantization")
        self.device_map_auto = QCheckBox("Device map auto")
        self.task_combo = QComboBox()
        self.task_combo.addItems(["VQA", "Image-to-Text"])
        self.max_image_size = QSpinBox()
        self.max_image_size.setRange(0, 4096)
        self.max_image_size.setSingleStep(128)
        self.max_image_size.setValue(0)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(1.0)
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(0.9)
        self.num_beams = QSpinBox()
        self.num_beams.setRange(1, 10)
        self.num_beams.setValue(1)
        self.max_new_tokens = QSpinBox()
        self.max_new_tokens.setRange(1, 512)
        self.max_new_tokens.setValue(32)
        
        # Simplified vision/memory inputs for chat interface
        self.vision_input = QLineEdit()
        self.memory_input = QLineEdit()
        self.vision_input.setVisible(False)  # Hidden in chat interface
        self.memory_input.setVisible(False)  # Hidden in chat interface
        
        # Create placeholder for results display (used by other methods)
        self.results_display = self.chat_history  # Point to chat history
        
        # Initialize model-aware UI hints on first load
        try:
            self.update_input_placeholder_for_model()
            self.update_model_hint()
        except Exception:
            pass
        
        return page

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
        """Update configuration options based on selected model."""
        model_id = self.local_model_path or self.model_combo.currentData()
        if not model_id:
            return
            
        # Model-specific configurations
        model_name = model_id.lower()
        
        # Update task combo based on model capabilities
        if hasattr(self, 'task_combo'):
            self.task_combo.clear()
            
            if "llava" in model_name or "onevision" in model_name:
                self.task_combo.addItems(["VQA", "Image-to-Text"])
                self.task_combo.setCurrentText("VQA")  # LLaVA is primarily for VQA
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
                # Default options
                self.task_combo.addItems(["VQA", "Image-to-Text"])
        
        # Update quantization recommendations
        if hasattr(self, 'use_4bit') and hasattr(self, 'use_8bit'):
            # Large models (>7B) benefit from quantization
            if any(size in model_name for size in ["7b", "8b", "13b", "large"]):
                if not self.use_4bit.isChecked() and not self.use_8bit.isChecked():
                    self.use_4bit.setChecked(True)  # Recommend 4-bit for large models
            
        # Update device mapping for large models
        if hasattr(self, 'device_map_auto'):
            if any(size in model_name for size in ["13b", "large"]) and torch.cuda.device_count() > 1:
                self.device_map_auto.setChecked(True)

    def update_input_placeholder_for_model(self):
        """Set chat input placeholder based on selected model capabilities."""
        try:
            model_id = self.local_model_path or self.model_combo.currentData()
        except Exception:
            model_id = None
        if not model_id:
            if hasattr(self, 'prompt_input'):
                self.prompt_input.setPlaceholderText("Ask a question about the image...")
            return

        name = str(model_id).lower()
        placeholder = "Ask a question about the image..."

        # Heuristics for model families
        if ("llava" in name) or ("onevision" in name) or ("vqa" in name) or ("vilt" in name):
            placeholder = "Ask a visual question, e.g., 'What is Pikachu doing?'"
        elif ("caption" in name) or ("image-caption" in name) or ("vit-gpt2" in name) or ("pix2struct" in name) or ("donut" in name) or ("git" in name):
            placeholder = "Describe the image, e.g., 'Generate a concise caption.'"
        elif ("chat" in name) or ("cogvlm" in name) or ("qwen-vl" in name):
            placeholder = "Chat with the image, e.g., 'Summarize the scene.'"

        if hasattr(self, 'prompt_input'):
            self.prompt_input.setPlaceholderText(placeholder)

    def update_model_hint(self):
        """Update a short usage hint on the chat tab based on the selected model."""
        if not hasattr(self, 'model_hint_label'):
            return
        try:
            model_id = self.local_model_path or self.model_combo.currentData()
        except Exception:
            model_id = None
        task = self.task_combo.currentText() if hasattr(self, 'task_combo') else "VQA"
        if not model_id:
            self.model_hint_label.setText("")
            return
        name = str(model_id).lower()

        # Base hint
        lines = []
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

        if hasattr(self, 'screens_combo') and self.screens_combo.currentText() in ("Two Screens", "Two"):
            lines.append("Note: Two Screens mode is active. Load both images for best results.")

        hint_text = " \u2022 ".join(lines)
        # Render as a readable single-line hint prefixed by a bullet for each item
        self.model_hint_label.setText(" • " + hint_text)

    def create_models_page(self):
        """Create the models page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        header = QLabel("Model Configuration")
        header.setObjectName("pageHeader")
        layout.addWidget(header)
        
        # Model selection form
        form_layout = QFormLayout()
        
        # Model dropdown (already created in chat page, but we need it here too)
        if not hasattr(self, 'model_combo'):
            self.model_combo = QComboBox()
            for model_id, model_name in self.available_models.items():
                self.model_combo.addItem(model_name, model_id)
        form_layout.addRow("Model:", self.model_combo)
        
        # Local model selection
        if not hasattr(self, 'local_model_btn'):
            self.local_model_btn = QPushButton("Browse Local Model...")
            self.local_model_btn.clicked.connect(self.pick_local_model)
        form_layout.addRow("Local Model:", self.local_model_btn)
        
        self.local_model_indicator = QLineEdit()
        self.local_model_indicator.setReadOnly(True)
        self.local_model_indicator.setPlaceholderText("No local model selected")
        form_layout.addRow("", self.local_model_indicator)
        
        
        # Task selection
        self.task_combo = QComboBox()
        self.task_combo.addItems(["VQA", "Image-to-Text"])
        form_layout.addRow("Task:", self.task_combo)
        
        layout.addLayout(form_layout)
        
        # Cache settings
        cache_group = QLabel("Cache Settings")
        cache_group.setObjectName("groupHeader")
        layout.addWidget(cache_group)
        
        cache_form = QFormLayout()
        self.cache_btn = QPushButton("Select Cache Directory...")
        self.cache_btn.clicked.connect(self.pick_cache_dir)
        cache_form.addRow("Cache Directory:", self.cache_btn)
        
        self.cache_indicator = QLineEdit()
        self.cache_indicator.setReadOnly(True)
        self.cache_indicator.setPlaceholderText("Default HF cache")
        cache_form.addRow("", self.cache_indicator)
        
        layout.addLayout(cache_form)
        
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

    def create_config_page(self):
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
        
        self.screens_combo = QComboBox()
        self.screens_combo.addItems(["One", "Two"])
        self.screens_combo.currentTextChanged.connect(self.on_screens_changed)
        img_form.addRow("Screens:", self.screens_combo)
        
        self.load_btn2 = QPushButton("Load Image 2")
        self.load_btn2.setEnabled(False)
        self.load_btn2.clicked.connect(self.load_image2)
        img_form.addRow("", self.load_btn2)
        
        self.image2_indicator = QLineEdit()
        self.image2_indicator.setReadOnly(True)
        self.image2_indicator.setPlaceholderText("No second image")
        img_form.addRow("", self.image2_indicator)
        
        layout.addLayout(img_form)
        
        # Advanced prompts
        prompt_group = QLabel("Advanced Prompts")
        prompt_group.setObjectName("groupHeader")
        layout.addWidget(prompt_group)
        
        layout.addWidget(QLabel("Vision Prompt (optional):"))
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

    def create_server_page(self):
        """Create the server page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        
        header = QLabel("Server Control")
        header.setObjectName("pageHeader")
        layout.addWidget(header)
        
        self.server_btn = QPushButton("🚀 Start Server")
        self.server_btn.clicked.connect(self.toggle_server)
        layout.addWidget(self.server_btn)
        
        layout.addStretch()
        return page

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
        """Load the selected model in the background."""
        model_name = self.local_model_path or self.model_combo.currentData()
        if not model_name:
            QMessageBox.warning(self, "No Model Selected", "Please select a model first.")
            return
        
        if self.model_load_worker and self.model_load_worker.isRunning():
            QMessageBox.information(self, "Loading in Progress", "A model is already being loaded.")
            return
        
        # Collect load options
        load_opts = {
            "use_8bit": self.use_8bit.isChecked(),
            "use_4bit": self.use_4bit.isChecked(),
            "device_map_auto": self.device_map_auto.isChecked(),
        }
        
        # Start loading
        self.model_load_worker = ModelLoader(model_name, self.cache_dir, load_opts)
        self.model_load_worker.finished.connect(self.on_model_loaded)
        self.model_load_worker.error.connect(self.on_model_load_error)
        self.model_load_worker.progress.connect(self.on_model_load_progress)
        
        # Update UI
        self.load_model_btn.setEnabled(False)
        self.model_status_label.setText("Loading model...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.model_load_worker.start()
    
    def on_model_loaded(self, pipeline_obj):
        """Handle successful model loading."""
        self.loaded_pipeline = pipeline_obj
        self.loaded_model_name = self.local_model_path or self.model_combo.currentData()
        
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
        self.progress_bar.setVisible(False)
        
        model_display_name = self.loaded_model_name
        if len(model_display_name) > 50:
            model_display_name = "..." + model_display_name[-47:]
        
        self.model_status_label.setText(f"✅ Loaded: {model_display_name}")
        
        QMessageBox.information(self, "Model Loaded", 
                               f"Model '{self.loaded_model_name}' loaded successfully!\n"
                               "Inference will now be much faster.")
    
    def on_model_load_error(self, error_msg):
        """Handle model loading error."""
        self.load_model_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.model_status_label.setText("❌ Load failed")
        
        QMessageBox.critical(self, "Model Load Error", f"Failed to load model:\n{error_msg}")
    
    def on_model_load_progress(self, value):
        """Update model loading progress."""
        self.progress_bar.setValue(value)
    
    def unload_model(self):
        """Unload the current model to free memory."""
        if self.loaded_pipeline:
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

    def config_matches_loaded_model(self):
        """Check if current configuration matches the loaded model's configuration."""
        if not self.loaded_model_config:
            return False
        
        current_config = {
            "use_8bit": self.use_8bit.isChecked(),
            "use_4bit": self.use_4bit.isChecked(),
            "device_map_auto": self.device_map_auto.isChecked(),
            "use_fp16": self.use_fp16.isChecked(),
        }
        
        return current_config == self.loaded_model_config

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

    def export_result(self):
        """Export the conversation (or latest answer) to a JSON file."""
        # Prepare export structure
        model_id = None
        try:
            model_id = self.local_model_path or self.model_combo.currentData()
        except Exception:
            pass

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
            "last_answer": self.last_answer if hasattr(self, 'last_answer') else None,
        }

        # If no conversation history yet, fallback to exporting just the last answer
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

    def on_toggle_4bit(self, checked: bool) -> None:
        if checked and self.use_8bit.isChecked():
            self.use_8bit.setChecked(False)

    def on_toggle_8bit(self, checked: bool) -> None:
        if checked and self.use_4bit.isChecked():
            self.use_4bit.setChecked(False)

    def init_statusbar(self) -> None:
        self.vram_label = QLabel("VRAM: n/a")
        self.time_label = QLabel("Time: n/a")
        self.server_label = QLabel("Server: Stopped")
        self.statusBar().addPermanentWidget(self.server_label)
        self.statusBar().addPermanentWidget(self.vram_label)
        self.statusBar().addPermanentWidget(self.time_label)

    def init_vram_timer(self) -> None:
        self.vram_timer = QTimer(self)
        self.vram_timer.setInterval(1000)
        self.vram_timer.timeout.connect(self.update_vram_status)
        self.vram_timer.start()

    def update_vram_status(self) -> None:
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                free_gb = free / (1024**3)
                total_gb = total / (1024**3)
                self.vram_label.setText(f"VRAM: {free_gb:.2f} / {total_gb:.2f} GB free")
            except Exception:
                self.vram_label.setText("VRAM: unknown")
        else:
            self.vram_label.setText("VRAM: CPU")

    def apply_styles(self):
        """Apply modern dark futuristic blue theme."""
        # Set application font
        font = QFont("Segoe UI", 9)
        self.setFont(font)
        
        # Modern dark blue futuristic theme
        self.setStyleSheet("""
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
            
            QLabel#sidebarHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #1d4ed8);
                color: white;
                font-size: 16pt;
                font-weight: bold;
                padding: 20px;
                margin: 0;
            }
            
            QPushButton#navButton {
                background: transparent;
                border: none;
                color: #94a3b8;
                font-size: 11pt;
                font-weight: 500;
                padding: 12px 16px;
                text-align: left;
                border-radius: 8px;
                margin: 2px 0;
            }
            
            QPushButton#navButton:hover {
                background: rgba(59, 130, 246, 0.1);
                color: #3b82f6;
            }
            
            QPushButton#navButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(59, 130, 246, 0.2), stop:1 rgba(59, 130, 246, 0.1));
                color: #60a5fa;
                border-left: 3px solid #3b82f6;
            }
            
            /* Page Headers */
            QLabel#pageHeader {
                color: #e2e8f0;
                font-size: 18pt;
                font-weight: bold;
                margin-bottom: 10px;
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
        """)
    
    def load_image(self):
        """Open a file dialog to let the user select an image to analyze."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.run_btn.setEnabled(True)

    def load_image2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select second image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if path:
            self.image2_path = path
            self.image2_indicator.setText(path)

    def pick_local_model(self):
        """Allow the user to select a local directory for a model."""
        path = QFileDialog.getExistingDirectory(self, "Select local model directory")
        if path:
            self.local_model_path = path
            self.local_model_indicator.setText(path)

    def pick_cache_dir(self):
        """Allow the user to select a cache directory for models."""
        path = QFileDialog.getExistingDirectory(self, "Select cache directory")
        if path:
            self.cache_dir = path
            self.cache_indicator.setText(path)
    
    def display_image(self, image_path):
        """Display the selected image scaled to fit within the preview area.

        Parameters
        ----------
        image_path : str
            Path to the image to display.
        """
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale the image to fit the label while maintaining aspect ratio
            pixmap = pixmap.scaled(
                max(100, self.image_label.width() - 20), 
                max(100, self.image_label.height() - 20),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    # Drag-and-drop support
    def dragEnterEvent(self, event):
        """Accept drag enter events if they contain supported image files."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if str(url.toLocalFile()).lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Handle dropped image files and display the first supported one."""
        if event.mimeData().hasUrls():
            local_files = [str(url.toLocalFile()) for url in event.mimeData().urls() if str(url.toLocalFile())]
            imgs = [p for p in local_files if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
            if imgs:
                self.current_image_path = imgs[0]
                self.display_image(imgs[0])
                if len(imgs) > 1:
                    self.image2_path = imgs[1]
                    self.image2_indicator.setText(self.image2_path)
                self.run_btn.setEnabled(True)
        event.acceptProposedAction()

    def resizeEvent(self, event):
        """Rescale the image preview responsively on window resize."""
        # Rescale image preview on window resize
        if self.current_image_path:
            self.display_image(self.current_image_path)
        super().resizeEvent(event)

    def run_vlm(self):
        """Run the selected VLM on the current image and user's question."""
        # Safely get prompt text regardless of whether prompt_input is QLineEdit or QTextEdit
        prompt_text = self.get_input_text(self.prompt_input).strip()
        if not self.current_image_path or not prompt_text:
            QMessageBox.warning(self, "Error", "Please load an image and enter a question.")
            return
        # Low VRAM warning
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                free_gb = free / (1024**3)
                if free_gb < 1.0:  # warn below 1 GB
                    resp = QMessageBox.question(
                        self,
                        "Low VRAM",
                        f"Only {free_gb:.2f} GB VRAM free. Continue anyway?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if resp != QMessageBox.Yes:
                        return
            except Exception:
                pass
        
        # Disable UI elements during processing
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get selected model and prompt
        model_id = self.local_model_path or self.model_combo.currentData()
        vision_text = self.get_input_text(self.vision_input).strip() if hasattr(self, 'vision_input') else ""
        memory_text = self.get_input_text(self.memory_input).strip() if hasattr(self, 'memory_input') else ""
        prompt = self.build_prompt(vision_text, memory_text, prompt_text)
        
        # Check if we have a pre-loaded model that matches
        use_preloaded = (self.loaded_pipeline is not None and 
                        self.loaded_model_name == model_id and
                        self.config_matches_loaded_model())
        
        # Warn user if model is loaded but config doesn't match
        if (self.loaded_pipeline is not None and 
            self.loaded_model_name == model_id and 
            not self.config_matches_loaded_model()):
            
            reply = QMessageBox.question(
                self, 
                "Configuration Changed", 
                "The loaded model was configured with different settings.\n\n"
                "Current settings (quantization, device mapping, etc.) don't match "
                "the loaded model configuration.\n\n"
                "Do you want to:\n"
                "• Continue with current loaded model (faster, but uses old settings)\n"
                "• Reload model with new settings (slower, but uses current config)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                # User wants to reload with new settings
                use_preloaded = False

        # Collect generation settings
        gen_params = {
            "answer_top_k": self.answer_top_k.value(),
            "temperature": self.temperature.value(),
            "top_p": self.top_p.value(),
            "num_beams": self.num_beams.value(),
            "max_new_tokens": self.max_new_tokens.value(),
        }
        
        # Preprocess options
        preprocess = {
            "use_fp16": bool(self.use_fp16.isChecked() and torch.cuda.is_available()),
            "max_image_size": int(self.max_image_size.value()),
        }
        load_opts = {
            "use_8bit": bool(self.use_8bit.isChecked()),
            "use_4bit": bool(self.use_4bit.isChecked()),
            "device_map_auto": bool(self.device_map_auto.isChecked()),
        }

        # If Two screens, merge images vertically
        pil_image = None
        if self.screens_combo.currentText() in ("Two Screens", "Two"):
            if not getattr(self, 'image2_path', None):
                QMessageBox.warning(self, "Error", "Please select the second image for Two screens.")
                self.set_ui_enabled(True)
                self.progress_bar.setVisible(False)
                return
            img1 = Image.open(self.current_image_path).convert("RGB")
            img2 = Image.open(self.image2_path).convert("RGB")
            w = max(img1.width, img2.width)
            h = img1.height + img2.height
            pil_image = Image.new("RGB", (w, h), (0, 0, 0))
            pil_image.paste(img1, (0, 0))
            pil_image.paste(img2, (0, img1.height))

        # Create and start worker thread
        self.worker = VLMWorker(
            model_id,
            self.current_image_path,
            prompt,
            cache_dir=self.cache_dir,
            gen_params=gen_params,
            preprocess=preprocess,
            pil_image=pil_image,
            load_opts=load_opts,
            task=self.task_combo.currentText(),
            preloaded_pipeline=self.loaded_pipeline if use_preloaded else None,
        )
        self.worker.finished.connect(self.on_vlm_finished)
        self.worker.error.connect(self.on_vlm_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.timing.connect(self.on_vlm_timing)
        self.worker.start()
    
    def update_progress(self, value):
        """Update the progress bar with the provided value."""
        self.progress_bar.setValue(value)
    
    def get_input_text(self, widget) -> str:
        """Return textual content from QLineEdit or QTextEdit uniformly."""
        # QLineEdit provides .text(), QTextEdit provides .toPlainText()
        if hasattr(widget, "text"):
            return widget.text()
        return widget.toPlainText()
    
    def on_vlm_finished(self, result):
        """Handle VLM inference completion with chat-style display."""
        # Get user message (supports QLineEdit or QTextEdit)
        user_message = self.get_input_text(self.prompt_input)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': self.get_timestamp()
        })
        self.conversation_history.append({
            'role': 'assistant', 
            'content': result,
            'timestamp': self.get_timestamp()
        })
        
        # Update chat display
        self.update_chat_display()
        
        # Clear input
        if hasattr(self.prompt_input, 'clear'):
            self.prompt_input.clear()
        else:
            self.prompt_input.setPlainText("")
        
        # Store last answer for export
        self.last_answer = result
        self.cleanup_worker()
    
    def get_timestamp(self):
        """Get current timestamp for chat messages."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M")
    
    def update_chat_display(self):
        """Update the chat history display."""
        if not hasattr(self, 'chat_history'):
            return
            
        chat_html = ""
        for msg in self.conversation_history:
            role = msg['role']
            content = msg['content']
            timestamp = msg['timestamp']
            
            if role == 'user':
                chat_html += f"""
                <div style="margin: 10px 0; text-align: right;">
                    <div style="background: #0084ff; color: white; padding: 8px 12px; border-radius: 18px; display: inline-block; max-width: 70%; word-wrap: break-word;">
                        {content}
                    </div>
                    <div style="font-size: 11px; color: #888; margin-top: 2px;">
                        You • {timestamp}
                    </div>
                </div>
                """
            else:
                chat_html += f"""
                <div style="margin: 10px 0; text-align: left;">
                    <div style="background: #f1f1f1; color: #333; padding: 8px 12px; border-radius: 18px; display: inline-block; max-width: 70%; word-wrap: break-word;">
                        {content}
                    </div>
                    <div style="font-size: 11px; color: #888; margin-top: 2px;">
                        Assistant • {timestamp}
                    </div>
                </div>
                """
        
        self.chat_history.setHtml(chat_html)
        
        # Scroll to bottom
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_vlm_timing(self, seconds: float) -> None:
        self.time_label.setText(f"Time: {seconds*1000:.0f} ms")
    
    def on_vlm_error(self, error_msg):
        """Display an error message dialog and reset the UI state."""
        QMessageBox.critical(self, "Error", error_msg)
        self.cleanup_worker()
    
    def cleanup_worker(self):
        """Tear down the worker thread and re-enable the UI."""
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None
        
        # Re-enable UI
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
    
    def set_ui_enabled(self, enabled):
        """Enable or disable interactive UI elements as a group."""
        self.load_btn.setEnabled(enabled)
        self.run_btn.setEnabled(enabled and bool(self.current_image_path))
        self.model_combo.setEnabled(enabled)
        self.prompt_input.setEnabled(enabled)
        self.local_model_btn.setEnabled(enabled)
        self.cache_btn.setEnabled(enabled)
        self.load_btn2.setEnabled(
            enabled and self.screens_combo.currentText() in ("Two Screens", "Two")
        )
    
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
        """Handle screen count change."""
        if hasattr(self, 'second_image_widget'):
            self.second_image_widget.setVisible(text == "Two Screens")
        if hasattr(self, 'load_btn2'):
            self.load_btn2.setEnabled(text == "Two Screens")
        # Refresh the model hint to reflect screen mode
        if hasattr(self, 'update_model_hint'):
            self.update_model_hint()

    def build_prompt(self, vision_text: str, memory_text: str, user_question: str) -> str:
        screens_line = (
            "[Input: Two gameplay screenshots]"
            if self.screens_combo.currentText() in ("Two Screens", "Two")
            else "[Input: One gameplay screenshot]"
        )
        return (
            f"{screens_line}\n\n"
            "Task: Analyze this Pokémon screenshot and respond ONLY with a single JSON object, no extra text. "
            "Player buttons: A, B, Up, Down, Left, Right, Start, Select, L, R.\n\n"
            "UI Modes:\n"
            "- BATTLE: HP bars, Pokémon sprites, battle interface\n"
            "- DIALOG: Text boxes\n"
            "- MENU: Menu lists, Pokémon stats/items\n"
            "- OVERWORLD: Player on map, no overlays\n\n"
            "Steps:\n"
            "1) Detect visible UI elements (HP bars, text boxes, menus, map)\n"
            "2) Pick the dominant screen mode\n"
            "3) If multiple modes present, pick the one requiring immediate player action\n\n"
            "JSON Schema:\n"
            "{\n"
            " \"ui_mode\": \"OVERWORLD|DIALOG|MENU|BATTLE\",\n"
            " \"visible_text\": \"concise text\",\n"
            " \"hp_estimates\": {\"player_hp_pct\":0..100,\"opponent_hp_pct\":0..100},\n"
            " \"menu_options\": [\"list of strings\"],\n"
            " \"suggested_action\": \"A|B|UP|DOWN|LEFT|RIGHT|START|SELECT|L|R\",\n"
            " \"reason\": \"brief, <10 words\",\n"
            " \"confidence\": 0.0..1.0\n"
            "}\n\n"
            "Rules:\n"
            "- Non-BATTLE → hp_estimates = {\"player_hp_pct\":0,\"opponent_hp_pct\":0}\n"
            "- Focus on player’s next required action\n"
            "- Default ui_mode=OVERWORLD if unsure\n"
            "- Default suggested_action=\"A\" with confidence=0.0 if unsure\n\n"
            f"Context:\nVISION_PROMPT: {vision_text}\nMEMORY: {memory_text}\n\n"
            "Return completed JSON object:\n"
            "{\n"
            " \"visible_text\": \"\",\n"
            " \"hp_estimates\": {\"player_hp_pct\":0,\"opponent_hp_pct\":0},\n"
            " \"menu_options\": [],\n"
            " \"suggested_action\": \"\",\n"
            " \"reason\": \"\",\n"
            " \"ui_mode\": \"\",\n"
            " \"confidence\": 0.0\n"
            "}\n\n"
            f"User Question: {user_question}\n"
        )

    def toggle_server(self):
        """Start or stop the FastAPI server."""
        if self.server_process is None:
            self.start_server()
        else:
            self.stop_server()

    def start_server(self):
        """Start the FastAPI server in a background thread."""
        try:
            # Start uvicorn server in a separate thread
            def run_server():
                self.server_process = subprocess.Popen([
                    sys.executable, "-m", "uvicorn", 
                    "app.server:app", 
                    "--host", "127.0.0.1", 
                    "--port", str(self.server_port)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.server_process.wait()
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            # Wait a moment for server to start
            import time
            time.sleep(2)
            
            # Test if server is actually running
            try:
                import requests
                response = requests.get(f"http://127.0.0.1:{self.server_port}/health", timeout=5)
                if response.status_code == 200:
                    # Update UI - server is confirmed running
                    self.server_btn.setText("🛑 Stop Server")
                    self.server_label.setText(f"Server: Running on http://127.0.0.1:{self.server_port}")
                    QMessageBox.information(
                        self, 
                        "Server Started", 
                        f"FastAPI server started on http://127.0.0.1:{self.server_port}\n\n"
                        "Your Pokémon application can now connect to:\n"
                        f"• GET http://127.0.0.1:{self.server_port}/health - Check server status\n"
                        f"• POST http://127.0.0.1:{self.server_port}/infer - Send images for analysis"
                    )
                else:
                    raise Exception("Server health check failed")
            except Exception:
                # Server didn't start properly
                self.stop_server()
                QMessageBox.critical(self, "Server Error", "Server failed to start properly. Check console for errors.")
                
        except Exception as e:
            QMessageBox.critical(self, "Server Error", f"Failed to start server: {e}")

    def stop_server(self):
        """Stop the FastAPI server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception:
                pass
            finally:
                self.server_process = None
        
        if self.server_thread:
            self.server_thread = None
        
        # Update UI
        self.server_btn.setText("🚀 Start Server")
        self.server_label.setText("Server: Stopped")

def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("VLM Explorer")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("VLM Explorer")
    
    # Set application icon globally
    icon_path = "D:/CascadeProjects/windsurf-project/favicon_io/favicon-32x32.png"
    try:
        app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass  # Fallback if icon not found
    
    # Set application style for better theming
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = VLMApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
