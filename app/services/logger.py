"""
Глобальный логгер приложения.
Логика полностью перенесена из backend.py без изменений.
"""

import logging

logger = logging.getLogger("deepfake_app")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# ---- Логи в файл ----
file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# ---- Логи в терминал ----
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

# ---- Добавляем обработчики ----
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("=== Приложение запущено ===")
