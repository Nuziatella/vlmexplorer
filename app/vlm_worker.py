"""VLM worker thread for running inference without blocking the UI.

Separated to keep GUI file size small and maintain modularity.
"""
from __future__ import annotations

import os
import contextlib
import time
import logging
from typing import Any, Dict, Optional

import torch
from PIL import Image
from PySide6.QtCore import QThread, Signal
from transformers import pipeline, BitsAndBytesConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vlm_explorer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_vlm_inference(
    *,
    model_name: str,
    prompt: str,
    cache_dir: Optional[str],
    gen_params: Dict[str, Any],
    preprocess: Dict[str, Any],
    pil_image: Optional[Image.Image],
    image_path: Optional[str],
    load_opts: Dict[str, Any],
    task: str,
) -> tuple[str, float]:
    """Run inference and return (answer_str, elapsed_seconds)."""
    logger.info(f"Starting VLM inference with model: {model_name}")
    logger.info(f"Task: {task}, Prompt length: {len(prompt)} chars")
    
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Using device: {'CUDA' if device != -1 else 'CPU'}")
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
    model_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    tokenizer_kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    dtype = torch.float16 if (device != -1 and preprocess.get("use_fp16", False)) else None
    # Quantization configuration (bitsandbytes)
    if device != -1:
        if load_opts.get("use_4bit"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif load_opts.get("use_8bit"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # Determine if this is a local model path or HF model ID
    is_local_model = os.path.exists(model_name) or (os.path.sep in model_name or '\\' in model_name)
    
    if is_local_model:
        logger.info(f"Detected local model path: {model_name}")
        # For local models, we need to specify the task since auto-detection requires HF Hub access
        # Try to infer from model name or default to image-to-text for most VLMs
        if "llava" in model_name.lower() or "onevision" in model_name.lower():
            pipe_task = "image-to-text"
        elif task == "Image-to-Text":
            pipe_task = "image-to-text"
        else:
            # Default to image-to-text for local VLMs as it's more versatile
            pipe_task = "image-to-text"
        logger.info(f"Using task '{pipe_task}' for local model")
    else:
        logger.info("Creating pipeline with automatic task detection for HF model")
        pipe_task = None  # Let HF auto-detect
    
    try:
        # Create pipeline kwargs
        pipeline_kwargs = {
            "task": pipe_task,
            "model": model_name,
            "dtype": dtype,
            "model_kwargs": model_kwargs,
            "tokenizer_kwargs": tokenizer_kwargs,
            "trust_remote_code": True,
        }
        
        # Only add device OR device_map, not both
        if load_opts.get("device_map_auto"):
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device
        
        vlm_pipeline = pipeline(**pipeline_kwargs)
        # Get the actual task that was used
        actual_task = vlm_pipeline.task
        logger.info(f"Pipeline created successfully with task: {actual_task}")
        pipe_task = actual_task
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise

    # Prepare image
    image = pil_image if pil_image is not None else Image.open(image_path).convert("RGB")
    max_dim = int(preprocess.get("max_image_size", 0))
    if max_dim and max_dim > 0:
        image.thumbnail((max_dim, max_dim))

    answer_top_k = int(gen_params.get("answer_top_k", 1))
    gen_kwargs: Dict[str, Any] = {}
    if "temperature" in gen_params:
        gen_kwargs["temperature"] = float(gen_params["temperature"])
    if "top_p" in gen_params:
        gen_kwargs["top_p"] = float(gen_params["top_p"])
    if "num_beams" in gen_params:
        gen_kwargs["num_beams"] = int(gen_params["num_beams"])
    if "max_new_tokens" in gen_params:
        gen_kwargs["max_new_tokens"] = int(gen_params["max_new_tokens"])

    logger.info("Starting inference...")
    start = time.perf_counter()
    try:
        with torch.inference_mode(), (
            torch.autocast("cuda") if (device != -1 and preprocess.get("use_fp16", False)) else contextlib.nullcontext()
        ):
            if pipe_task == "image-to-text":
                # For image-to-text models (including LLaVA), format prompt appropriately
                if task == "VQA" and prompt:
                    # Format as a conversation for VQA-style questions
                    full_prompt = f"USER: {prompt}\nASSISTANT:"
                else:
                    full_prompt = prompt if prompt else ""
                result = vlm_pipeline(image, prompt=full_prompt, generate_kwargs=gen_kwargs or None)
            elif pipe_task == "visual-question-answering":
                # For dedicated VQA models
                result = vlm_pipeline(
                    image=image,
                    question=prompt,
                    top_k=answer_top_k,
                    generate_kwargs=gen_kwargs or None,
                )
            else:
                # For other auto-detected tasks, try the most generic approach
                logger.info(f"Using generic approach for task: {pipe_task}")
                result = vlm_pipeline(image, generate_kwargs=gen_kwargs or None)
    except TypeError as e:
        logger.warning(f"TypeError during inference, trying without generate_kwargs: {e}")
        with torch.inference_mode(), (
            torch.autocast("cuda") if (device != -1 and preprocess.get("use_fp16", False)) else contextlib.nullcontext()
        ):
            if pipe_task == "image-to-text":
                if task == "VQA" and prompt:
                    full_prompt = f"USER: {prompt}\nASSISTANT:"
                else:
                    full_prompt = prompt if prompt else ""
                result = vlm_pipeline(image, prompt=full_prompt)
            elif pipe_task == "visual-question-answering":
                result = vlm_pipeline(image=image, question=prompt, top_k=answer_top_k)
            else:
                result = vlm_pipeline(image)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise
    
    elapsed = time.perf_counter() - start
    logger.info(f"Inference completed in {elapsed:.3f} seconds")

    # Normalize result to answer string
    if isinstance(result, list) and result and isinstance(result[0], dict) and "answer" in result[0]:
        answer = str(result[0]["answer"])
    else:
        answer = str(result)
    return answer, elapsed


class ModelLoader(QThread):
    """Background thread for loading VLM models without blocking the UI."""
    
    finished = Signal(object)  # Emits the loaded pipeline
    error = Signal(str)
    progress = Signal(int)
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None, 
                 load_opts: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.load_opts = load_opts or {"use_8bit": False, "use_4bit": False, "device_map_auto": False}
        self._is_running = True
    
    def stop(self):
        self._is_running = False
    
    def run(self):
        """Load the model pipeline."""
        logger.info(f"ModelLoader starting for model: {self.model_name}")
        try:
            self.progress.emit(25)
            
            # Set up model loading parameters
            device = 0 if torch.cuda.is_available() else -1
            if self.cache_dir:
                os.environ["HF_HOME"] = self.cache_dir
            
            model_kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
            tokenizer_kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
            
            # Add quantization if specified
            if device != -1:
                if self.load_opts.get("use_4bit"):
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                elif self.load_opts.get("use_8bit"):
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            
            self.progress.emit(50)
            
            # Determine task and create pipeline
            is_local_model = os.path.exists(self.model_name) or (os.path.sep in self.model_name or '\\' in self.model_name)
            
            self.progress.emit(75)
            
            # Special handling for LLaVA-OneVision models
            if "llava" in self.model_name.lower() and "onevision" in self.model_name.lower():
                logger.info("Detected LLaVA-OneVision model, using direct model loading")
                try:
                    from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
                    from transformers import AutoConfig
                    
                    # First, try to load the config to check for custom components
                    try:
                        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                        logger.info(f"Model config loaded successfully: {type(config)}")
                    except Exception as config_error:
                        logger.warning(f"Config loading failed: {config_error}")
                        raise config_error
                    
                    # Load model and processor directly
                    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                        self.model_name,
                        device_map=("auto" if self.load_opts.get("device_map_auto") else None),
                        dtype=torch.float16 if (device != -1) else None,
                        trust_remote_code=True,
                        **model_kwargs
                    )
                    processor = LlavaOnevisionProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        **tokenizer_kwargs
                    )
                    
                    # Create a custom pipeline-like object
                    class LlavaOnevisionPipeline:
                        def __init__(self, model, processor):
                            self.model = model
                            self.processor = processor
                            self.task = "image-to-text"
                        
                        def __call__(self, image, prompt="", **kwargs):
                            # Format the conversation
                            conversation = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image"},
                                        {"type": "text", "text": prompt}
                                    ]
                                }
                            ]
                            
                            # Apply chat template
                            prompt_text = self.processor.apply_chat_template(
                                conversation, add_generation_prompt=True
                            )
                            
                            # Process inputs
                            inputs = self.processor(
                                text=prompt_text,
                                images=image,
                                return_tensors="pt"
                            )
                            
                            # Move to device
                            if device != -1:
                                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                            
                            # Generate
                            with torch.no_grad():
                                output_ids = self.model.generate(
                                    **inputs,
                                    max_new_tokens=kwargs.get("generate_kwargs", {}).get("max_new_tokens", 512),
                                    temperature=kwargs.get("generate_kwargs", {}).get("temperature", 1.0),
                                    do_sample=kwargs.get("generate_kwargs", {}).get("temperature", 1.0) > 0,
                                )
                            
                            # Decode response
                            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                            response = self.processor.batch_decode(
                                generated_ids, skip_special_tokens=True
                            )[0]
                            
                            return [{"generated_text": response}]
                    
                    pipeline_obj = LlavaOnevisionPipeline(model, processor)
                    
                except (ImportError, KeyError, Exception) as e:
                    logger.warning(f"Direct LLaVA-OneVision loading failed ({e}), trying alternative approaches")
                    
                    # Try with AutoModel approach
                    try:
                        from transformers import AutoModelForCausalLM, AutoProcessor
                        logger.info("Trying AutoModel approach for LLaVA-OneVision")
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            device_map=("auto" if self.load_opts.get("device_map_auto") else None),
                            dtype=torch.float16 if (device != -1) else None,
                            trust_remote_code=True,
                            **model_kwargs
                        )
                        processor = AutoProcessor.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            **tokenizer_kwargs
                        )
                        
                        # Create a generic LLaVA pipeline
                        class GenericLlavaPipeline:
                            def __init__(self, model, processor):
                                self.model = model
                                self.processor = processor
                                self.task = "image-to-text"
                            
                            def __call__(self, image, prompt="", **kwargs):
                                try:
                                    # Try the conversation format first
                                    if hasattr(self.processor, 'apply_chat_template'):
                                        conversation = [
                                            {
                                                "role": "user",
                                                "content": [
                                                    {"type": "image"},
                                                    {"type": "text", "text": prompt}
                                                ]
                                            }
                                        ]
                                        prompt_text = self.processor.apply_chat_template(
                                            conversation, add_generation_prompt=True
                                        )
                                    else:
                                        # Fallback to simple prompt
                                        prompt_text = f"USER: {prompt}\nASSISTANT:"
                                    
                                    # Process inputs
                                    inputs = self.processor(
                                        text=prompt_text,
                                        images=image,
                                        return_tensors="pt"
                                    )
                                    
                                    # Move to device
                                    if hasattr(self.model, 'device'):
                                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                                    
                                    # Generate
                                    with torch.no_grad():
                                        output_ids = self.model.generate(
                                            **inputs,
                                            max_new_tokens=kwargs.get("generate_kwargs", {}).get("max_new_tokens", 512),
                                            temperature=kwargs.get("generate_kwargs", {}).get("temperature", 1.0),
                                            do_sample=kwargs.get("generate_kwargs", {}).get("temperature", 1.0) > 0,
                                            pad_token_id=self.processor.tokenizer.eos_token_id,
                                        )
                                    
                                    # Decode response
                                    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                                    response = self.processor.batch_decode(
                                        generated_ids, skip_special_tokens=True
                                    )[0]
                                    
                                    return [{"generated_text": response}]
                                    
                                except Exception as inner_e:
                                    logger.error(f"Error in GenericLlavaPipeline: {inner_e}")
                                    return [{"generated_text": f"Error: {inner_e}"}]
                        
                        pipeline_obj = GenericLlavaPipeline(model, processor)
                        
                    except Exception as auto_error:
                        logger.warning(f"AutoModel approach also failed ({auto_error}), falling back to standard pipeline")
                        # Final fallback to standard pipeline
                        fallback_kwargs = {
                            "task": "image-to-text",
                            "model": self.model_name,
                            "dtype": torch.float16 if (device != -1) else None,
                            "model_kwargs": model_kwargs,
                            "tokenizer_kwargs": tokenizer_kwargs,
                            "trust_remote_code": True,
                        }
                        
                        # Only add device OR device_map, not both
                        if self.load_opts.get("device_map_auto"):
                            fallback_kwargs["device_map"] = "auto"
                        else:
                            fallback_kwargs["device"] = device
                        
                        pipeline_obj = pipeline(**fallback_kwargs)
            else:
                # Standard pipeline loading for other models
                if is_local_model:
                    if "llava" in self.model_name.lower():
                        pipe_task = "image-to-text"
                    else:
                        pipe_task = "image-to-text"  # Default for local VLMs
                else:
                    pipe_task = None  # Auto-detect for HF models
                
                # Create the pipeline
                pipeline_kwargs = {
                    "task": pipe_task,
                    "model": self.model_name,
                    "dtype": torch.float16 if (device != -1) else None,
                    "model_kwargs": model_kwargs,
                    "tokenizer_kwargs": tokenizer_kwargs,
                    "trust_remote_code": True,
                }
                
                # Only add device OR device_map, not both
                if self.load_opts.get("device_map_auto"):
                    pipeline_kwargs["device_map"] = "auto"
                else:
                    pipeline_kwargs["device"] = device
                
                pipeline_obj = pipeline(**pipeline_kwargs)
            
            if not self._is_running:
                return
            
            self.progress.emit(100)
            self.finished.emit(pipeline_obj)
            logger.info("ModelLoader completed successfully")
            
        except Exception as e:
            error_msg = f"ModelLoader error: {e}"
            logger.error(error_msg, exc_info=True)
            self.error.emit(error_msg)


class VLMWorker(QThread):
    """Background thread that runs VLM inference without blocking the UI.

    Parameters
    ----------
    model_name : str | os.PathLike
        HF model ID or local directory path.
    image_path : str
        Original image path (still passed for reference); may be unused when pil_image provided.
    prompt : str
        Text prompt to supply to the model.
    cache_dir : Optional[str]
        HF cache directory.
    gen_params : Optional[Dict[str, Any]]
        Generation params such as temperature, top_p, num_beams, max_new_tokens, answer_top_k.
    preprocess : Optional[Dict[str, Any]]
        Preprocess options: {use_fp16: bool, max_image_size: int}
    pil_image : Optional[Image.Image]
        If provided, used directly instead of loading from image_path.
    load_opts : Optional[Dict[str, Any]]
        Loading options: {use_8bit: bool, device_map_auto: bool}
    task : str
        Either "VQA" or "Image-to-Text".
    """

    finished = Signal(str)
    timing = Signal(float)
    progress = Signal(int)
    error = Signal(str)

    def __init__(
        self,
        model_name: str,
        image_path: str,
        prompt: str,
        *,
        cache_dir: Optional[str] = None,
        gen_params: Optional[Dict[str, Any]] = None,
        preprocess: Optional[Dict[str, Any]] = None,
        pil_image: Optional[Image.Image] = None,
        load_opts: Optional[Dict[str, Any]] = None,
        task: str = "VQA",
        preloaded_pipeline: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.image_path = image_path
        self.prompt = prompt
        self.cache_dir = cache_dir
        self.gen_params = gen_params or {}
        self.preprocess = preprocess or {"use_fp16": False, "max_image_size": 0}
        self.pil_image = pil_image
        self.load_opts = load_opts or {"use_8bit": False, "use_4bit": False, "device_map_auto": False}
        self.task = task
        self.preloaded_pipeline = preloaded_pipeline
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False

    def run_preloaded_inference(self) -> tuple[str, float]:
        """Run inference using the preloaded pipeline."""
        import time
        import contextlib
        
        # Prepare image
        image = self.pil_image if self.pil_image is not None else Image.open(self.image_path).convert("RGB")
        max_dim = int(self.preprocess.get("max_image_size", 0))
        if max_dim and max_dim > 0:
            image.thumbnail((max_dim, max_dim))

        # Prepare generation parameters
        gen_kwargs = {}
        if "temperature" in self.gen_params:
            gen_kwargs["temperature"] = float(self.gen_params["temperature"])
        if "top_p" in self.gen_params:
            gen_kwargs["top_p"] = float(self.gen_params["top_p"])
        if "num_beams" in self.gen_params:
            gen_kwargs["num_beams"] = int(self.gen_params["num_beams"])
        if "max_new_tokens" in self.gen_params:
            gen_kwargs["max_new_tokens"] = int(self.gen_params["max_new_tokens"])

        answer_top_k = int(self.gen_params.get("answer_top_k", 1))
        pipe_task = self.preloaded_pipeline.task
        device = 0 if torch.cuda.is_available() else -1

        logger.info("Starting preloaded inference...")
        start = time.perf_counter()
        
        try:
            with torch.inference_mode(), (
                torch.autocast("cuda") if (device != -1 and self.preprocess.get("use_fp16", False)) else contextlib.nullcontext()
            ):
                # Check if this is our custom LlavaOnevisionPipeline or GenericLlavaPipeline
                if hasattr(self.preloaded_pipeline, 'processor') and hasattr(self.preloaded_pipeline, 'model'):
                    # Custom LLaVA pipeline (OneVision or Generic)
                    result = self.preloaded_pipeline(image, prompt=self.prompt, generate_kwargs=gen_kwargs)
                elif pipe_task == "image-to-text":
                    # For image-to-text models, format prompt appropriately
                    if self.task == "VQA" and self.prompt:
                        full_prompt = f"USER: {self.prompt}\nASSISTANT:"
                    else:
                        full_prompt = self.prompt if self.prompt else ""
                    result = self.preloaded_pipeline(image, prompt=full_prompt, generate_kwargs=gen_kwargs or None)
                elif pipe_task == "visual-question-answering":
                    # For dedicated VQA models
                    result = self.preloaded_pipeline(
                        image=image,
                        question=self.prompt,
                        top_k=answer_top_k,
                        generate_kwargs=gen_kwargs or None,
                    )
                else:
                    # For other tasks, try generic approach
                    result = self.preloaded_pipeline(image, generate_kwargs=gen_kwargs or None)
        except TypeError as e:
            logger.warning(f"TypeError during preloaded inference, trying without generate_kwargs: {e}")
            with torch.inference_mode(), (
                torch.autocast("cuda") if (device != -1 and self.preprocess.get("use_fp16", False)) else contextlib.nullcontext()
            ):
                # Check if this is our custom LlavaOnevisionPipeline or GenericLlavaPipeline
                if hasattr(self.preloaded_pipeline, 'processor') and hasattr(self.preloaded_pipeline, 'model'):
                    # Custom LLaVA pipeline (no generate_kwargs)
                    result = self.preloaded_pipeline(image, prompt=self.prompt)
                elif pipe_task == "image-to-text":
                    if self.task == "VQA" and self.prompt:
                        full_prompt = f"USER: {self.prompt}\nASSISTANT:"
                    else:
                        full_prompt = self.prompt if self.prompt else ""
                    result = self.preloaded_pipeline(image, prompt=full_prompt)
                elif pipe_task == "visual-question-answering":
                    result = self.preloaded_pipeline(image=image, question=self.prompt, top_k=answer_top_k)
                else:
                    result = self.preloaded_pipeline(image)
        
        elapsed = time.perf_counter() - start
        logger.info(f"Preloaded inference completed in {elapsed:.3f} seconds")

        # Normalize result to answer string
        if isinstance(result, list) and result and isinstance(result[0], dict):
            if "generated_text" in result[0]:
                answer = str(result[0]["generated_text"])
            elif "answer" in result[0]:
                answer = str(result[0]["answer"])
            else:
                answer = str(result[0])
        else:
            answer = str(result)
        
        return answer, elapsed

    def run(self) -> None:  # noqa: D401
        """Execute the VLM pipeline on the provided image and prompt."""
        logger.info(f"VLMWorker starting for model: {self.model_name}")
        try:
            self.progress.emit(50)
            
            if self.preloaded_pipeline:
                logger.info("Using preloaded pipeline for faster inference")
                answer, elapsed = self.run_preloaded_inference()
            else:
                logger.info("Loading model and running inference")
                answer, elapsed = run_vlm_inference(
                    model_name=self.model_name,
                    prompt=self.prompt,
                    cache_dir=self.cache_dir,
                    gen_params=self.gen_params,
                    preprocess=self.preprocess,
                    pil_image=self.pil_image,
                    image_path=self.image_path,
                    load_opts=self.load_opts,
                    task=self.task,
                )
            if not self._is_running:
                logger.info("VLMWorker stopped by user")
                return
            self.progress.emit(100)
            self.timing.emit(float(elapsed))
            self.finished.emit(answer)
            logger.info("VLMWorker completed successfully")
        except Exception as e:  # noqa: BLE001
            error_msg = f"VLMWorker error: {e}"
            logger.error(error_msg, exc_info=True)
            self.error.emit(error_msg)
