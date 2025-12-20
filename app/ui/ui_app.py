from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import QStackedWidget
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import QTimer


from PIL import Image as PILImage

from app.core.inference import DeepfakeClassifier, PredictionResult
from app.services.logger import logger
from app.core.video import is_video_path, read_video_uniform_frames
from app.core.inference import VideoPredictionResult

class MainWindow(QMainWindow):
    def __init__(self, classifier: DeepfakeClassifier):
        super().__init__()

        self.classifier = classifier
        self.current_image_path: Optional[Path] = None
        self.current_pil_image: Optional[PILImage.Image] = None

        self.setWindowTitle("Deepfake Detector — SigLIP2")
        self.resize(1100, 700)
        self.setAcceptDrops(True)

        logger.info("GUI запущено.")

        self._init_ui()

        self.media_player = QMediaPlayer(self)
        self.media_player.mediaStatusChanged.connect(self.on_media_status)

        self.media_player.setVideoOutput(self.video_widget)

        self.media_player.errorOccurred.connect(self._on_player_error)
        self.media_player.mediaStatusChanged.connect(self._on_media_status)
        self.media_player.playbackStateChanged.connect(self._on_playback_state)

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ================== LEFT PANEL — IMAGE =======================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #000000;
                border-radius: 10px;
            }
        """)

        from PyQt6.QtWidgets import QSizePolicy
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("""
            background-color: #FFFFFF;
            border: none;
        """)

        self.preview_stack = QStackedWidget()
        self.preview_stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.preview_stack.setAutoFillBackground(True)

        self.preview_stack.setStyleSheet("""
            QStackedWidget {
                background-color: #FFFFFF;
                border: 1px solid #000000;
                border-radius: 10px;
                padding: 2px;   /* КЛЮЧЕВО */
            }
        """)

        self.preview_stack.addWidget(self.image_label)   # index 0
        self.preview_stack.addWidget(self.video_widget)  # index 1

        left_layout.addWidget(self.preview_stack)


        # ================== RIGHT PANEL — CONTROLS ======================
        right_panel = QWidget()
        right_panel.setMinimumWidth(380)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(14)

        # ----- Controls -----
        controls_box = QGroupBox("")
        controls_box.setStyleSheet(
            """
            QGroupBox {
                color: #ddd;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
            }
            """
        )

        controls_layout = QGridLayout(controls_box)
        controls_layout.setContentsMargins(12, 12, 12, 12)

        self.open_btn = QPushButton("Открыть файл")
        self.open_btn.clicked.connect(self.open_image)
        self.open_btn.setMinimumHeight(36)
        controls_layout.addWidget(self.open_btn, 0, 0, 1, 2)

        controls_layout.addWidget(QLabel("Порог (%):"), 1, 0)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setSuffix(" %")
        self.threshold_spin.setValue(50)
        self.threshold_spin.setMinimumHeight(30)
        controls_layout.addWidget(self.threshold_spin, 1, 1)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setValue(50)
        controls_layout.addWidget(self.threshold_slider, 2, 0, 1, 2)

        self.predict_btn = QPushButton("Анализировать")
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setMinimumHeight(40)
        controls_layout.addWidget(self.predict_btn, 3, 0, 1, 2)

        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setMinimumHeight(32)
        self.clear_btn.clicked.connect(self.clear_interface)
        controls_layout.addWidget(self.clear_btn, 4, 0, 1, 2)

        # Sync порога
        self.threshold_spin.valueChanged.connect(self.threshold_slider.setValue)
        self.threshold_slider.valueChanged.connect(self.threshold_spin.setValue)

        # ----- Results -----
        result_box = QGroupBox("")
        result_box.setStyleSheet(
            """
            QGroupBox {
                color: #ddd;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
            }
            """
        )

        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(12, 12, 12, 12)

        self.result_label = QLabel("Нет данных.")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.result_label.setStyleSheet("color: #000; font-size: 13px;")
        self.result_label.setMinimumHeight(120)
        result_layout.addWidget(self.result_label)

        # ----- Final UI assemble -----
        right_layout.addWidget(controls_box)
        right_layout.addWidget(result_box, stretch=1)

        main_layout.addWidget(left_panel, stretch=4)
        main_layout.addWidget(right_panel, stretch=2)

        # Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("color: white;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готово.")

        # Применить общий стиль (если хочешь — можно вызвать здесь)
        self._apply_style()

        # ----- Video controls -----
        video_controls = QHBoxLayout()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setVisible(False)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_play_pause)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_video)

        video_controls.addWidget(self.play_btn)
        video_controls.addWidget(self.stop_btn)

        right_layout.addLayout(video_controls)

    # -------------------------------------------------------------------------
    # СТИЛИ
    # -------------------------------------------------------------------------

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
            }

            QLabel {
                color: #000000;
                font-size: 14px;
            }

            /* ГРУППОВЫЕ БЛОКИ */
            QGroupBox {
                background-color: #FFFFFF;
                color: #000000;
                border: 1px solid #000000;   /* ЧЁРНАЯ рамка */
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 8px;
            }

            /* КНОПКИ */
            QPushButton {
                background-color:#3A7AFE;
                color:white;
                border:none;
                border-radius:6px;
                padding:8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color:#2F64D3;
            }

            /* SPINBOX */
            QSpinBox {
                background: #FFFFFF;
                color: #000000;
                border: 1px solid #000000;    /* ЧЁРНАЯ рамка */
                border-radius: 4px;
                padding: 2px;
            }

            /* SLIDER */
            QSlider::groove:horizontal {
                border: 1px solid #000000;
                height: 6px;
                background: #E0E0E0;
                margin: 0px;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #3A7AFE;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #E0E0E0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3A7AFE;
                border: 1px solid #000000;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }

            /* PROGRESS BAR – БЕЗ ТЕКСТА */
            QProgressBar {
                border: 1px solid #000000;   /* ЧЁРНАЯ рамка */
                background: #FFFFFF;         /* БЕЛЫЙ фон */
                border-radius: 6px;
                color: transparent;          /* УБИРАЕМ ТЕКСТ */
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3A7AFE;
                border-radius: 6px;
            }

            QStatusBar {
                background-color: #FFFFFF;
                color: #000000;
            }
        """)

        # Чёрная рамка вокруг изображения
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: none;
            }
        """)

        self.video_widget.setStyleSheet("""
            background-color: #FFFFFF;
            border: none;
        """)

    # ------------------------------------------------------------------------
    # ЛОГИКА
    # -------------------------------------------------------------------------

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

        self.predict_btn.setEnabled(True)   # ← ВОТ ЭТОГО НЕ ХВАТАЛО

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

    def run_prediction(self):
        if self.current_media_path is None:
            return

        logger.info("Пользователь запустил анализ.")
        threshold = self.threshold_spin.value() / 100
        self.status_bar.showMessage("Анализ...")

        try:
            if self.current_media_type == "video":
                result = self.classifier.predict_video(
                    self.current_media_path,
                    threshold=threshold,
                    agg_method="median_of_means",
                    chunk_count=8,
                    batch_size=16,
                    max_side=768,
                )
                self._display_video_result(result)
            else:
                # image
                if self.current_pil_image is None:
                    return
                result = self.classifier.predict(self.current_pil_image, threshold)
                self._display_result(result)

        except Exception as e:
            logger.error(f"Ошибка инференса: {e}")
            self.status_bar.showMessage("Ошибка анализа!")
            return

    def _display_result(self, result: PredictionResult):
        prob_pct = int(result.prob_deepfake * 100)
        label_ru = "ДИПФЕЙК" if result.label == "deepfake" else "НАСТОЯЩЕЕ"

        self.result_label.setText(
            f"<b>{label_ru}</b><br>"
            f"Вероятность дипфейка: <b>{result.prob_deepfake:.4f}</b> ({prob_pct}%)"
        )

        self.status_bar.showMessage("Готово.")

        logger.info("Вывод результата завершён.")

    def _display_video_result(self, result: VideoPredictionResult):
        prob_pct = int(result.prob_deepfake * 100)
        label_ru = "ДИПФЕЙК" if result.label == "deepfake" else "НАСТОЯЩЕЕ"

        n = len(result.per_frame_probs)
        if n > 0:
            p_min = min(result.per_frame_probs)
            p_max = max(result.per_frame_probs)
            p_mean = sum(result.per_frame_probs) / n
        else:
            p_min = p_max = p_mean = 0.0

        self.result_label.setText(
            f"<b>{label_ru}</b><br>"
            f"Итоговая вероятность дипфейка: <b>{result.prob_deepfake:.4f}</b> ({prob_pct}%)<br><br>"
            f"Метод агрегации: <b>{result.agg_method}</b><br>"
            f"Длительность: <b>{result.meta.duration_sec:.1f}s</b>, FPS: <b>{result.meta.fps:.2f}</b><br>"
            f"Сэмплов кадров: <b>{n}</b><br>"
            f"Статистика по кадрам: mean={p_mean:.3f}, min={p_min:.3f}, max={p_max:.3f}"
        )

        self.status_bar.showMessage("Готово.")

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            suffix = path.suffix.lower()

            image_exts = {
                ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".jfif"
            }
            video_exts = {
                ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp"
            }

            if suffix in image_exts:
                logger.info(f"Перетащено изображение: {path}")
                self.load_image_from_path(path)
                return

            if suffix in video_exts:
                logger.info(f"Перетащено видео: {path}")
                self._load_video(path)
                return

    def clear_interface(self):
        self.media_player.stop()
        self.preview_stack.setCurrentIndex(0)

        self.image_label.clear()
        self.image_label.setText("Изображение не загружено")

        self.current_image_path = None
        self.current_pil_image = None
        self.current_media_path = None
        self.current_media_type = None

        self.play_btn.setVisible(False)
        self.stop_btn.setVisible(False)

        self.predict_btn.setEnabled(False)
        self.status_bar.showMessage("Очищено.")
    
    def dragEnterEvent(self, event):
        if not event.mimeData().hasUrls():
            return

        for url in event.mimeData().urls():
            suffix = Path(url.toLocalFile()).suffix.lower()
            if suffix in {
                ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".jfif",
                ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp",
            }:
                event.acceptProposedAction()
                return
            
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
    


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    classifier = DeepfakeClassifier()
    window = MainWindow(classifier)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
