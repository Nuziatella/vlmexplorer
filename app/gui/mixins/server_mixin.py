import sys
import subprocess
import threading
import time
import os

import requests
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame
from PySide6.QtCore import QTimer


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

        self.server_btn = QPushButton("START SERVER")
        self.server_btn.clicked.connect(self.toggle_server)
        self.server_btn.setObjectName("primaryButton")
        layout.addWidget(self.server_btn)

        # Status and metrics area
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")
        status_layout = QVBoxLayout(status_frame)
        
        self.server_label = QLabel("Server: Stopped")
        self.server_label.setObjectName("serverStatus")
        status_layout.addWidget(self.server_label)
        layout.addWidget(status_frame)

        # Metrics panel with futuristic styling
        metrics_frame = QFrame()
        metrics_frame.setObjectName("metricsFrame")
        metrics_layout = QVBoxLayout(metrics_frame)
        
        # Metrics header
        metrics_header = QLabel("SYSTEM METRICS")
        metrics_header.setObjectName("metricsHeader")
        metrics_layout.addWidget(metrics_header)
        
        # Metrics rows
        metrics_row1 = QHBoxLayout()
        self.metrics_in_label = QLabel("Incoming: 0")
        self.metrics_in_label.setObjectName("metricLabel")
        self.metrics_out_label = QLabel("Succeeded: 0 (Failed: 0)")
        self.metrics_out_label.setObjectName("metricLabel")
        metrics_row1.addWidget(self.metrics_in_label)
        metrics_row1.addWidget(self.metrics_out_label)
        metrics_layout.addLayout(metrics_row1)

        metrics_row2 = QHBoxLayout()
        self.metrics_latency_label = QLabel("Latency (avg/last): - / - ms")
        self.metrics_latency_label.setObjectName("metricLabel")
        self.metrics_uptime_label = QLabel("Uptime: 0.0 s")
        self.metrics_uptime_label.setObjectName("metricLabel")
        metrics_row2.addWidget(self.metrics_latency_label)
        metrics_row2.addWidget(self.metrics_uptime_label)
        metrics_layout.addLayout(metrics_row2)
        
        layout.addWidget(metrics_frame)

        # Start a timer placeholder (created on server start)
        self.server_metrics_timer = None

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
                # Ensure 'app.server:app' can be imported by setting CWD to project root
                # server_mixin.py is at app/gui/mixins/, so go up 4 levels
                project_root = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))
                        )
                    )
                )
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
                    cwd=project_root,
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
                    self.server_btn.setText("STOP SERVER")
                    self.server_label.setText(
                        f"Server: Running on http://127.0.0.1:{self.server_port}"
                    )
                    # Start polling metrics periodically
                    self.start_metrics_polling()
                    
                    # Show success message
                    QMessageBox.information(
                        self,
                        "Server Started",
                        (
                            f"FastAPI server started on http://127.0.0.1:{self.server_port}\n\n"
                            "Your Pokémon application can now connect to:\n"
                            f"• GET http://127.0.0.1:{self.server_port}/health - Check server status\n"
                            f"• POST http://127.0.0.1:{self.server_port}/infer - Send images for analysis\n"
                            f"• GET http://127.0.0.1:{self.server_port}/metrics - Server counters and latency"
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

        # Stop metrics polling
        self.stop_metrics_polling()

        # Update UI
        self.server_btn.setText("START SERVER")
        self.server_label.setText("Server: Stopped")
        self.metrics_in_label.setText("Incoming: 0")
        self.metrics_out_label.setText("Succeeded: 0 (Failed: 0)")
        self.metrics_latency_label.setText("Latency (avg/last): - / - ms")
        self.metrics_uptime_label.setText("Uptime: 0.0 s")

    # ------------------
    # Metrics helpers
    # ------------------
    def start_metrics_polling(self):
        """Begin polling /metrics periodically and updating labels."""
        if self.server_metrics_timer is None:
            self.server_metrics_timer = QTimer(self)
            self.server_metrics_timer.setInterval(2000)  # 2 seconds
            self.server_metrics_timer.timeout.connect(self.update_server_status)
            self.server_metrics_timer.start()

    def stop_metrics_polling(self):
        """Stop polling /metrics."""
        if self.server_metrics_timer is not None:
            try:
                self.server_metrics_timer.stop()
            except Exception:
                pass
            self.server_metrics_timer = None

    def update_server_status(self):
        """Fetch /metrics and /health to update server labels."""
        try:
            base = f"http://127.0.0.1:{self.server_port}"
            # Metrics
            r = requests.get(f"{base}/metrics", timeout=1.0)
            if r.status_code == 200 and isinstance(r.json(), dict):
                m = r.json()
                self.metrics_in_label.setText(f"Incoming: {m.get('incoming', 0)}")
                self.metrics_out_label.setText(
                    f"Succeeded: {m.get('succeeded', 0)} (Failed: {m.get('failed', 0)})"
                )
                avg = m.get('avg_latency_ms')
                last = m.get('last_latency_ms')
                avg_s = f"{avg:.1f}" if isinstance(avg, (int, float)) else "-"
                last_s = f"{last:.1f}" if isinstance(last, (int, float)) else "-"
                self.metrics_latency_label.setText(f"Latency (avg/last): {avg_s} / {last_s} ms")
                up = m.get('uptime_seconds', 0.0)
                self.metrics_uptime_label.setText(f"Uptime: {float(up):.1f} s")

            # Health (optional) to set a quick status; do not block if metrics already OK
            try:
                h = requests.get(f"{base}/health", timeout=1.0)
                if h.status_code == 200:
                    self.server_label.setText(f"Server: Running on {base}")
            except Exception:
                pass

        except Exception:
            # If metrics fetch fails, keep previous values; optionally mark server as stopped
            self.server_label.setText("Server: Unreachable")
