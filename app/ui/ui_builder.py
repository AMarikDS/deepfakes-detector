from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QStackedWidget,
)

from PyQt6.QtMultimediaWidgets import QVideoWidget


def build_ui(self):
    central = QWidget()
    self.setCentralWidget(central)

    main_layout = QHBoxLayout(central)
    main_layout.setContentsMargins(10, 10, 10, 10)
    main_layout.setSpacing(12)

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

    right_panel = QWidget()
    right_panel.setMinimumWidth(380)
    right_layout = QVBoxLayout(right_panel)
    right_layout.setSpacing(14)

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

    self.threshold_spin.valueChanged.connect(self.threshold_slider.setValue)
    self.threshold_slider.valueChanged.connect(self.threshold_spin.setValue)

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

    right_layout.addWidget(controls_box)
    right_layout.addWidget(result_box, stretch=1)

    main_layout.addWidget(left_panel, stretch=4)
    main_layout.addWidget(right_panel, stretch=2)

    self.status_bar = QStatusBar()
    self.status_bar.setStyleSheet("color: white;")
    self.setStatusBar(self.status_bar)
    self.status_bar.showMessage("Готово.")

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
