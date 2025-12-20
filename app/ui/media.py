from pathlib import Path

from PIL import Image as PILImage
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QPixmap
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import QFileDialog

from app.core.video import is_video_path, read_video_uniform_frames
from app.services.logger import logger


def open_image(self):
    dialog = QFileDialog(self, "Выберите файл")
    dialog.setNameFilter(
        "Файлы (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif *.jfif "
        "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mpg *.mpeg *.3gp)"
    )

    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return

    paths = dialog.selectedFiles()
    if not paths:
        return

    path = Path(paths[0])
    logger.info(f"Пользователь выбрал файл: {path}")

    self.load_media_from_path(path)


def load_media_from_path(self, path: Path):
    if is_video_path(path):
        self._load_video(path)
    else:
        self.load_image_from_path(path)


def load_image_from_path(self, path: Path):
    try:
        img = PILImage.open(path)
    except Exception as e:
        logger.error(f"Ошибка открытия изображения: {e}")
        self.status_bar.showMessage("Ошибка загрузки.")
        return

    self.current_image_path = path
    self.current_pil_image = img.copy()

    self.current_media_path = path
    self.current_media_type = "image"

    self.media_player.stop()
    self.preview_stack.setCurrentIndex(0)

    self._update_preview(path)

    self.play_btn.setVisible(False)
    self.stop_btn.setVisible(False)

    self.predict_btn.setEnabled(True)
    self.status_bar.showMessage("Изображение загружено.")

    self.play_btn.setEnabled(False)
    self.stop_btn.setEnabled(False)
    self.play_btn.setText("▶ Play")


def _load_video(self, path: Path):
    self.current_media_path = path
    self.current_media_type = "video"

    self.media_player.stop()
    self.preview_stack.setCurrentIndex(1)

    self.video_widget.show()
    self.video_widget.repaint()

    self.media_player.setSource(QUrl.fromLocalFile(str(path)))
    self.media_player.setPosition(0)

    self.play_btn.setVisible(True)
    self.stop_btn.setVisible(True)

    self.play_btn.setEnabled(True)
    self.stop_btn.setEnabled(True)
    self.play_btn.setText("▶ Play")

    self.predict_btn.setEnabled(True)  # ← ВОТ ЭТОГО НЕ ХВАТАЛО

    self.status_bar.showMessage("Видео загружено. Нажмите Play.")


def _update_preview(self, path: Path):
    pix = QPixmap(str(path))
    if pix.isNull():
        self.image_label.setText("Не удалось показать изображение.")
        return

    scaled = pix.scaled(
        self.image_label.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    self.image_label.setPixmap(scaled)


def clear_interface(self):
    # Остановить видео
    self.media_player.stop()
    self.preview_stack.setCurrentIndex(0)

    # Очистить превью
    self.image_label.clear()
    self.image_label.setText("Изображение не загружено")

    # 🔹 ОЧИСТИТЬ РЕЗУЛЬТАТ
    self.result_label.clear()
    self.result_label.setText("Нет данных.")

    # Сброс состояния
    self.current_image_path = None
    self.current_pil_image = None
    self.current_media_path = None
    self.current_media_type = None

    # Кнопки видео
    self.play_btn.setVisible(False)
    self.stop_btn.setVisible(False)

    # Кнопка анализа
    self.predict_btn.setEnabled(False)

    self.status_bar.showMessage("Очищено.")


def _on_player_error(self, error, error_string):
    logger.error(f"QMediaPlayer error: {error} | {error_string}")
    self.status_bar.showMessage(f"Ошибка видео: {error_string}")


def _on_media_status(self, status):
    logger.info(f"QMediaPlayer mediaStatusChanged: {status}")


def _on_playback_state(self, state):
    logger.info(f"QMediaPlayer playbackStateChanged: {state}")


def toggle_play_pause(self):
    state = self.media_player.playbackState()

    if state == QMediaPlayer.PlaybackState.PlayingState:
        self.media_player.pause()
        self.play_btn.setText("▶ Play")
    else:
        self.media_player.play()
        self.play_btn.setText("⏸ Pause")


def stop_video(self):
    self.media_player.stop()
    self.media_player.setPosition(0)
    self.play_btn.setText("▶ Play")


def on_media_status(self, status):
    if status == QMediaPlayer.MediaStatus.EndOfMedia:
        self.media_player.stop()
        self.media_player.setPosition(0)
        self.play_btn.setText("▶ Play")
