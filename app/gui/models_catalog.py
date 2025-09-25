AVAILABLE_MODELS = {
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
