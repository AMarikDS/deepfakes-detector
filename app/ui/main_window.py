from __future__ import annotations

import sys
from typing import Optional
from pathlib import Path

from PIL import Image as PILImage
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import QMainWindow

from app.core.inference import DeepfakeClassifier
from app.services.logger import logger

from app.ui.ui_builder import build_ui
from app.ui.styles import apply_style
from app.ui import media as media_ops
from app.ui import inference_ui as inference_ops
from app.ui import drag_drop as dnd_ops


class MainWindow(QMainWindow):
    def __init__(self, classifier: DeepfakeClassifier):
        super().__init__()

        self.classifier = classifier
        self.current_image_path: Optional[Path] = None
        self.current_pil_image: Optional[PILImage.Image] = None
        self.current_media_path: Optional[Path] = None
        self.current_media_type: Optional[str] = None

        self.setWindowTitle("Deepfake Detector — SigLIP2")
        self.resize(1100, 700)
        self.setAcceptDrops(True)

        logger.info("GUI запущено.")

        build_ui(self)
        apply_style(self)

        self.media_player = QMediaPlayer(self)
        self.media_player.mediaStatusChanged.connect(self.on_media_status)

        self.media_player.setVideoOutput(self.video_widget)

        self.media_player.errorOccurred.connect(self._on_player_error)
        self.media_player.mediaStatusChanged.connect(self._on_media_status)
        self.media_player.playbackStateChanged.connect(self._on_playback_state)

    # -------------------------------------------------------------------------
    # ПРОКСИ-МЕТОДЫ: имена и поведение как в исходном файле
    # -------------------------------------------------------------------------

    # --- UI actions ---
    def open_image(self):
        media_ops.open_image(self)

    def clear_interface(self):
        media_ops.clear_interface(self)

    # --- Media load ---
    def load_image_from_path(self, path: Path):
        media_ops.load_image_from_path(self, path)

    def load_media_from_path(self, path: Path):
        media_ops.load_media_from_path(self, path)

    def _load_video(self, path: Path):
        media_ops._load_video(self, path)

    def _update_preview(self, path: Path):
        media_ops._update_preview(self, path)

    # --- Inference ---
    def run_prediction(self):
        inference_ops.run_prediction(self)

    def _display_result(self, result):
        inference_ops._display_result(self, result)

    def _display_video_result(self, result):
        inference_ops._display_video_result(self, result)

    # --- Drag & Drop ---
    def dragEnterEvent(self, event):
        dnd_ops.dragEnterEvent(self, event)

    def dropEvent(self, event):
        dnd_ops.dropEvent(self, event)

    # --- Player callbacks ---
    def _on_player_error(self, error, error_string):
        media_ops._on_player_error(self, error, error_string)

    def _on_media_status(self, status):
        media_ops._on_media_status(self, status)

    def _on_playback_state(self, state):
        media_ops._on_playback_state(self, state)

    def toggle_play_pause(self):
        media_ops.toggle_play_pause(self)

    def stop_video(self):
        media_ops.stop_video(self)

    def on_media_status(self, status):
        media_ops.on_media_status(self, status)


# -----------------------------------------------------------------------------
# MAIN (оставляем только в app/ui/app.py, здесь не нужен)
# -----------------------------------------------------------------------------
