def apply_style(self):
    self.setStyleSheet("""
        QMainWindow {
            background-color: #FFFFFF;
        }

        QLabel {
            color: #000000;
            font-size: 14px;
        }

        QGroupBox {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #000000;
            border-radius: 8px;
            margin-top: 14px;
            padding-top: 8px;
        }

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

        QSpinBox {
            background: #FFFFFF;
            color: #000000;
            border: 1px solid #000000;
            border-radius: 4px;
            padding: 2px;
        }

        QSlider::groove:horizontal {
            border: 1px solid #000000;
            height: 6px;
            background: #E0E0E0;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #3A7AFE;
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

        QProgressBar {
            border: 1px solid #000000;
            background: #FFFFFF;
            border-radius: 6px;
            color: transparent;
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
