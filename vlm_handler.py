from typing import Optional
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class VLMHandler:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the Vision-Language Model handler.
        
        Args:
            model_name: Name of the BLIP model to use from Hugging Face
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def generate_caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Generate a caption for the given image with an optional prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Optional text prompt to guide the generation
            
        Returns:
            Generated caption text
        """
        try:
            # Load and process the image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            if prompt:
                inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            out = self.model.generate(**inputs, max_length=100)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            return f"Error generating caption: {str(e)}"
