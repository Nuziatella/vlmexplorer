import sys
import subprocess
import threading
import time
import os
import socket
import importlib

import requests
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame, QPlainTextEdit
from PySide6.QtCore import QTimer, QObject, Signal


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

        # Read-only console for server stdout/stderr
        self.server_console = QPlainTextEdit()
        self.server_console.setObjectName("serverConsole")
        self.server_console.setReadOnly(True)
        self.server_console.setMinimumHeight(220)
        # Subtle styling to match theme
        self.server_console.setStyleSheet(
            """
            QPlainTextEdit#serverConsole {
                background: #0c0e14;
                color: #c9d4e3;
                border: 1px solid #2a385a;
                border-radius: 6px;
                padding: 6px;
            }
            """
        )
        layout.addWidget(self.server_console)

        # Emitter to safely append lines to console from background threads
        class _LogEmitter(QObject):
            line = Signal(str)

        self._log_emitter = _LogEmitter()
        self._log_emitter.line.connect(self.server_console.appendPlainText)

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
            # Clear console and print startup banner
            try:
                if hasattr(self, "server_console") and self.server_console is not None:
                    self.server_console.clear()
                    self.server_console.appendPlainText(
                        f"[INFO] Preparing to start server on port {self.server_port}..."
                    )
            except Exception:
                pass
            
            # Compute project root and build command
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )
            )
            cmd_preview = [
                sys.executable,
                "-m",
                "uvicorn",
                "app.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.server_port),
                "--log-level",
                "info",
            ]
            try:
                self._log_emitter.line.emit(
                    f"[INFO] Launch cmd: {' '.join(cmd_preview)} (cwd={project_root})"
                )
            except Exception:
                pass

            # Preflight: ensure app.server importable
            try:
                importlib.import_module("app.server")
                try:
                    self._log_emitter.line.emit("[INFO] Preflight import app.server OK")
                except Exception:
                    pass
            except Exception as e:
                try:
                    self._log_emitter.line.emit(f"[ERROR] Preflight import failed: {e}")
                except Exception:
                    pass
                QMessageBox.critical(
                    self,
                    "Server Error",
                    f"Cannot import backend module 'app.server':\n{e}\n\nCheck PYTHONPATH and package layout.",
                )
                return

            # Preflight: check port availability
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("127.0.0.1", int(self.server_port)))
                sock.close()
                try:
                    self._log_emitter.line.emit(f"[INFO] Port {self.server_port} appears available")
                except Exception:
                    pass
            except OSError as e:
                try:
                    self._log_emitter.line.emit(
                        f"[ERROR] Port {self.server_port} is in use or not bindable: {e}"
                    )
                except Exception:
                    pass
                QMessageBox.critical(
                    self,
                    "Port In Use",
                    (
                        f"Port {self.server_port} is unavailable. Close the other process or change the port "
                        "in the code (self.server_port)."
                    ),
                )
                return
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
                cmd = [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.server:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.server_port),
                    "--log-level",
                    "info",
                ]
                # Log the command used
                try:
                    self._log_emitter.line.emit(
                        f"[INFO] Starting server: {' '.join(cmd)} (cwd={project_root})"
                    )
                except Exception:
                    pass

                # Ensure project root is on PYTHONPATH for module imports
                env = os.environ.copy()
                env["PYTHONPATH"] = (
                    project_root + os.pathsep + env.get("PYTHONPATH", "")
                    if project_root not in env.get("PYTHONPATH", "")
                    else env.get("PYTHONPATH", "")
                )

                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=project_root,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                # Start background readers for stdout/stderr
                self._start_log_threads()
                self.server_process.wait()

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

            # Wait for server to start (retry health check for up to ~10s)
            ok = False
            for _ in range(20):
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{self.server_port}/health", timeout=1.0
                    )
                    if response.status_code == 200:
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)

            # Test if server is actually running
            try:
                if ok:
                    # Update UI - server is confirmed running
                    self.server_btn.setText("STOP SERVER")
                    self.server_label.setText(
                        f"Server: Running on http://127.0.0.1:{self.server_port}"
                    )
                    try:
                        self._log_emitter.line.emit("[INFO] Server is healthy and running.")
                    except Exception:
                        pass
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
                    # Log return code if process exited
                    rc = None
                    try:
                        if getattr(self, "server_process", None) is not None:
                            rc = self.server_process.poll()
                    except Exception:
                        rc = None
                    try:
                        self._log_emitter.line.emit(
                            f"[ERROR] Health check failed. Process returncode={rc}"
                        )
                    except Exception:
                        pass
                    raise Exception("Server health check failed")
            except Exception:
                # Server didn't start properly
                self.stop_server()
                try:
                    self._log_emitter.line.emit("[ERROR] Server failed to start. See logs above.")
                except Exception:
                    pass
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

        # Stop log reader threads
        self._stop_log_threads()

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

        try:
            self._log_emitter.line.emit("[INFO] Server stopped.")
        except Exception:
            pass

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

    # ------------------
    # Log thread helpers
    # ------------------
    def _start_log_threads(self):
        try:
            self._server_log_stop = threading.Event()
            self._stdout_thread = threading.Thread(
                target=self._read_stream, args=(self.server_process.stdout, "STDOUT"), daemon=True
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stream, args=(self.server_process.stderr, "STDERR"), daemon=True
            )
            self._stdout_thread.start()
            self._stderr_thread.start()
        except Exception:
            pass

    def _stop_log_threads(self):
        try:
            if hasattr(self, "_server_log_stop") and self._server_log_stop is not None:
                self._server_log_stop.set()
            for attr in ("_stdout_thread", "_stderr_thread"):
                t = getattr(self, attr, None)
                if t:
                    try:
                        t.join(timeout=1.0)
                    except Exception:
                        pass
                    setattr(self, attr, None)
        except Exception:
            pass

    def _read_stream(self, stream, label: str):
        try:
            while stream and not getattr(self, "_server_log_stop", threading.Event()).is_set():
                line = stream.readline()
                if not line:
                    break
                line = line.rstrip("\r\n")
                try:
                    self._log_emitter.line.emit(f"[{label}] {line}")
                except Exception:
                    pass
        except Exception:
            pass
