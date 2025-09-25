from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel
import torch


class StatusBarMixin:
    """Provides status bar and VRAM updates."""

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
