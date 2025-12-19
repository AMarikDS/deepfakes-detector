"""
Конфигурация путей, устройств и dtype.
Логика полностью взята из backend.py, ничего не изменено.
"""

import os
import torch
from PIL import Image as PILImage
from PIL import ImageFile as PILImageFile

from app.services.logger import logger


# -----------------------------------------------------------------------------
# ПУТИ
# -----------------------------------------------------------------------------

PROJECT_DIR = "./models/siglip2_deepfake-diffusion-full"
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoint-657")
BASE_MODEL_ID = CKPT_DIR


# -----------------------------------------------------------------------------
# УСТРОЙСТВО
# -----------------------------------------------------------------------------

def detect_device() -> str:
    if torch.backends.mps.is_available():
        logger.info("MPS доступен — используется GPU Apple Silicon")
        return "mps"
    if torch.cuda.is_available():
        logger.info("CUDA доступна — используется NVIDIA GPU")
        return "cuda"
    logger.info("Используется CPU")
    return "cpu"


def select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


DEVICE = detect_device()
DTYPE = select_dtype(DEVICE)
logger.info(f"DEVICE={DEVICE}, DTYPE={DTYPE}")

# настройки PIL
PILImageFile.LOAD_TRUNCATED_IMAGES = True
PILImage.MAX_IMAGE_PIXELS = None
