"""Debug script to manually start the FastAPI server and check for issues."""
import subprocess
import sys
import time
import requests

def start_server_debug():
    """Start the server with visible output for debugging."""
    port = 8001  # Use 8001 to avoid Windows port conflicts
    print("🚀 Starting FastAPI server in debug mode...")
    print(f"   Command: uvicorn app.server:app --host 127.0.0.1 --port {port}")
    print("   Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Start server with visible output
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.server:app", 
            "--host", "127.0.0.1", 
            "--port", str(port),
            "--reload"  # Enable auto-reload for development
        ])
        
        # Wait a moment then test
        print("⏳ Waiting 3 seconds for server to start...")
        time.sleep(3)
        
        print("🔍 Testing health endpoint...")
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is responding!")
                data = response.json()
                print(f"   Status: {data['status']}")
                print(f"   CUDA: {data['cuda']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        print(f"\n🌐 Server should be running at: http://127.0.0.1:{port}")
        print(f"📖 API docs available at: http://127.0.0.1:{port}/docs")
        print("\nPress Ctrl+C to stop the server...")
        
        # Wait for user to stop
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server_debug()
