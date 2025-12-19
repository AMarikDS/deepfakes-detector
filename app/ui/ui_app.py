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

from PIL import Image as PILImage

from app.core.inference import DeepfakeClassifier, PredictionResult
from app.services.logger import logger


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

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            """
            QLabel {
                background-color: #111;
                border: 1px solid #333;
                border-radius: 8px;
            }
            """
        )
        from PyQt6.QtWidgets import QSizePolicy

        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        left_layout.addWidget(self.preview_label)

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
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #000000;
                border-radius: 10px;
            }
        """)

        
    # -------------------------------------------------------------------------
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

        self._update_preview(path)
        self.predict_btn.setEnabled(True)
        self.status_bar.showMessage("Изображение загружено.")

    def open_image(self):
        dialog = QFileDialog(self, "Выберите изображение")
        dialog.setNameFilter(
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif *.jfif)"
        )

        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return

        paths = dialog.selectedFiles()
        if not paths:
            return

        path = Path(paths[0])
        logger.info(f"Пользователь выбрал файл: {path}")

        self.load_image_from_path(path)

    def _update_preview(self, path: Path):
        pix = QPixmap(str(path))
        if pix.isNull():
            self.preview_label.setText("Не удалось показать изображение.")
            return

        scaled = pix.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def run_prediction(self):
        if self.current_pil_image is None:
            return

        logger.info("Пользователь запустил анализ.")

        threshold = self.threshold_spin.value() / 100
        self.status_bar.showMessage("Анализ...")

        try:
            result = self.classifier.predict(self.current_pil_image, threshold)
        except Exception as e:
            logger.error(f"Ошибка инференса: {e}")
            self.status_bar.showMessage("Ошибка анализа!")
            return

        self._display_result(result)

    def _display_result(self, result: PredictionResult):
        prob_pct = int(result.prob_deepfake * 100)
        label_ru = "ДИПФЕЙК" if result.label == "deepfake" else "НАСТОЯЩЕЕ"

        self.result_label.setText(
            f"<b>{label_ru}</b><br>"
            f"Вероятность дипфейка: <b>{result.prob_deepfake:.4f}</b> ({prob_pct}%)"
        )

        self.status_bar.showMessage("Готово.")

        logger.info("Вывод результата завершён.")

    def clear_interface(self):
        logger.info("Кнопка 'Очистить' нажата.")

        self.preview_label.clear()
        self.preview_label.setText("Изображение не загружено")
        self.result_label.setText("Нет данных.")

        self.current_image_path = None
        self.current_pil_image = None

        self.predict_btn.setEnabled(False)
        self.status_bar.showMessage("Очищено.")

    # -------------------------------------------------------------------------
    # DRAG & DROP
    # -------------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".webp",
                ".tiff",
                ".jfif",
            ]:
                logger.info(f"Файл перетащен: {path}")
                self.load_image_from_path(path)
                break


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
