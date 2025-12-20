from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication

from app.core.inference import DeepfakeClassifier
from app.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    classifier = DeepfakeClassifier()
    window = MainWindow(classifier)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
