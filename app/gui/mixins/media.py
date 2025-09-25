from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog


class MediaMixin:
    """Image I/O, drag-and-drop, and responsive preview scaling."""

    def load_image(self) -> None:
        """Open a file chooser and load the selected image into the preview.

        On success, updates `self.current_image_path`, displays the image in
        `self.image_label`, and enables `self.run_btn` if present.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)",
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            if hasattr(self, "run_btn"):
                self.run_btn.setEnabled(True)

    def load_image2(self) -> None:
        """Pick a second image (for Two Screens mode) and update indicators."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select second image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if path:
            self.image2_path = path
            if hasattr(self, "image2_indicator"):
                self.image2_indicator.setText(path)
            if hasattr(self, "cfg_image2_indicator"):
                self.cfg_image2_indicator.setText(path)

    def pick_local_model(self) -> None:
        """Choose a local directory as the model source and reflect it in UI."""
        path = QFileDialog.getExistingDirectory(self, "Select local model directory")
        if path:
            self.local_model_path = path
            if hasattr(self, "local_model_indicator"):
                self.local_model_indicator.setText(path)

    def display_image(self, image_path: str) -> None:
        """Render the image at `image_path` scaled appropriately in the preview."""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                max(100, self.image_label.width() - 20),
                max(100, self.image_label.height() - 20),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # Drag-and-drop support
    def dragEnterEvent(self, event):  # type: ignore[override]
        """Accept drags when at least one URL points to a supported image file."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if str(url.toLocalFile()).lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):  # type: ignore[override]
        """Load the first dropped image and optionally set a second when present."""
        if event.mimeData().hasUrls():
            local_files = [str(url.toLocalFile()) for url in event.mimeData().urls() if str(url.toLocalFile())]
            imgs = [p for p in local_files if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
            if imgs:
                self.current_image_path = imgs[0]
                self.display_image(imgs[0])
                if len(imgs) > 1:
                    self.image2_path = imgs[1]
                    if hasattr(self, "image2_indicator"):
                        self.image2_indicator.setText(self.image2_path)
                    if hasattr(self, "cfg_image2_indicator"):
                        self.cfg_image2_indicator.setText(self.image2_path)
                if hasattr(self, "run_btn"):
                    self.run_btn.setEnabled(True)
        event.acceptProposedAction()

    def resizeEvent(self, event):  # type: ignore[override]
        """Keep the preview scaled to the label size on window resize."""
        if getattr(self, "current_image_path", None):
            self.display_image(self.current_image_path)
        super().resizeEvent(event)
