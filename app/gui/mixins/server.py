import sys
import subprocess
import threading
import time

import requests
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox


class ServerPageMixin:
    """Provides server page UI and control methods."""

    def create_server_page(self) -> QWidget:
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

    def toggle_server(self):
        """Start or stop the FastAPI server."""
        if getattr(self, "server_process", None) is None:
            self.start_server()
        else:
            self.stop_server()

    def start_server(self):
        """Start the FastAPI server in a background thread."""
        try:
            # Start uvicorn server in a separate thread
            def run_server():
                self.server_process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "app.server:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(self.server_port),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.server_process.wait()

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

            # Wait a moment for server to start
            time.sleep(2)

            # Test if server is actually running
            try:
                response = requests.get(
                    f"http://127.0.0.1:{self.server_port}/health", timeout=5
                )
                if response.status_code == 200:
                    # Update UI - server is confirmed running
                    self.server_btn.setText("🛑 Stop Server")
                    if hasattr(self, "server_label"):
                        self.server_label.setText(
                            f"Server: Running on http://127.0.0.1:{self.server_port}"
                        )
                    QMessageBox.information(
                        self,
                        "Server Started",
                        (
                            f"FastAPI server started on http://127.0.0.1:{self.server_port}\n\n"
                            "Your Pokémon application can now connect to:\n"
                            f"• GET http://127.0.0.1:{self.server_port}/health - Check server status\n"
                            f"• POST http://127.0.0.1:{self.server_port}/infer - Send images for analysis"
                        ),
                    )
                else:
                    raise Exception("Server health check failed")
            except Exception:
                # Server didn't start properly
                self.stop_server()
                QMessageBox.critical(
                    self,
                    "Server Error",
                    "Server failed to start properly. Check console for errors.",
                )

        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Server Error", f"Failed to start server: {e}")

    def stop_server(self):
        """Stop the FastAPI server."""
        if getattr(self, "server_process", None):
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception:
                pass
            finally:
                self.server_process = None

        if getattr(self, "server_thread", None):
            self.server_thread = None

        # Update UI
        self.server_btn.setText("🚀 Start Server")
        if hasattr(self, "server_label"):
            self.server_label.setText("Server: Stopped")
