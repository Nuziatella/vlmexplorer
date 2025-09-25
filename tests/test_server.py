"""Test script to verify FastAPI server integration.

This script demonstrates how your Pokémon application can connect to the VLM server.
"""
import base64
import requests
from PIL import Image
import io

def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def test_server_health():
    """Test the /health endpoint."""
    print("🔍 Checking server health...")
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server Health Check:")
            print(f"   Status: {data['status']}")
            print(f"   CUDA Available: {data['cuda']}")
            if data.get('vram') and isinstance(data['vram'], dict):
                print(f"   VRAM: {data['vram']['free_gb']:.2f} / {data['vram']['total_gb']:.2f} GB")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused - Server not running.")
        print("   💡 Make sure to:")
        print("   1. Run: python main.py")
        print("   2. Click 'Start Server' button")
        print("   3. Wait for 'Server Started' confirmation")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout - Server may be starting up.")
        print("   💡 Wait a few more seconds and try again.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_inference_example():
    """Test the /infer endpoint with a sample request."""
    # Create a simple test image (you can replace this with an actual Pokemon screenshot)
    test_image = Image.new("RGB", (256, 192), color="blue")  # DS resolution
    buffer = io.BytesIO()
    test_image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Sample request matching your Pokemon JSON template
    request_data = {
        "model_name": "Salesforce/blip-vqa-base",
        "task": "VQA",
        "prompt": """[Input: One gameplay screenshot]

Task: Analyze this Pokémon screenshot and respond ONLY with a single JSON object, no extra text. Player buttons: A, B, Up, Down, Left, Right, Start, Select, L, R.

UI Modes:
- BATTLE: HP bars, Pokémon sprites, battle interface
- DIALOG: Text boxes
- MENU: Menu lists, Pokémon stats/items
- OVERWORLD: Player on map, no overlays

JSON Schema:
{
 "ui_mode": "OVERWORLD|DIALOG|MENU|BATTLE",
 "visible_text": "concise text",
 "hp_estimates": {"player_hp_pct":0..100,"opponent_hp_pct":0..100},
 "menu_options": ["list of strings"],
 "suggested_action": "A|B|UP|DOWN|LEFT|RIGHT|START|SELECT|L|R",
 "reason": "brief, <10 words",
 "confidence": 0.0..1.0
}

User Question: What should I do next?""",
        "load_opts": {
            "use_4bit": True,
            "use_8bit": False,
            "device_map_auto": False
        },
        "preprocess": {
            "use_fp16": True,
            "max_image_size": 0
        },
        "image_b64_top": image_b64,
        "image_b64_bottom": None
    }
    
    try:
        print("\n🔄 Testing inference...")
        response = requests.post(
            "http://127.0.0.1:8001/infer",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Inference successful:")
            print(f"   Answer: {result['answer']}")
            print(f"   Time: {result['elapsed_seconds']:.3f} seconds")
            return True
        else:
            print(f"❌ Inference failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out (model loading can take time)")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Run server tests."""
    print("🧪 VLM Server Integration Test")
    print("=" * 40)
    
    # Test health endpoint
    if not test_server_health():
        print("\n💡 To start the server:")
        print("   1. Run: python main.py")
        print("   2. Click 'Start Server' button")
        print("   3. Run this test again")
        return
    
    # Test inference endpoint
    test_inference_example()
    
    print("\n✨ Integration test complete!")
    print("\n📖 For your Pokémon app:")
    print("   • Use the same request format as shown above")
    print("   • Send base64-encoded screenshots to /infer")
    print("   • Parse the JSON response for game actions")

if __name__ == "__main__":
    main()
