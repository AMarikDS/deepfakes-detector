"""
Классификатор DeepfakeClassifier.
Полностью перенесён из backend.py.
"""

from dataclasses import dataclass

import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor

from app.services.logger import logger
from app.config.settings import DEVICE, DTYPE, CKPT_DIR, BASE_MODEL_ID
from app.core.model import DeepfakeSigLIP
from app.core.model_loader import load_weights_from_checkpoint
from app.core.preprocess import normalize_image_to_rgb


@dataclass
class PredictionResult:
    label: str
    prob_deepfake: float
    confidence: float


class DeepfakeClassifier:
    def __init__(self):
        logger.info("Инициализация DeepfakeClassifier...")
        logger.info(f"Model path: {BASE_MODEL_ID}")

        self.device = DEVICE
        self.dtype = DTYPE

        logger.info("Загрузка AutoImageProcessor...")
        self.processor = AutoImageProcessor.from_pretrained(CKPT_DIR)
        logger.info("Processor загружен.")

        logger.info("Создание DeepfakeSigLIP модели...")
        self.model = DeepfakeSigLIP(BASE_MODEL_ID).to(self.device, dtype=self.dtype)

        logger.info("Загрузка весов...")
        load_weights_from_checkpoint(self.model, CKPT_DIR)

        logger.info("DeepfakeClassifier инициализирован.")

    @torch.no_grad()
    def predict(self, image: PILImage.Image, threshold=0.5) -> PredictionResult:
        logger.info(f"=== ЗАПУСК ИНФЕРЕНСА, threshold={threshold} ===")

        img = normalize_image_to_rgb(image)
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float16)

        logger.info("Прогон через модель...")
        logits = self.model(pixel_values)
        prob = torch.sigmoid(logits.float()).item()

        label = "deepfake" if prob >= threshold else "real"
        confidence = prob if label == "deepfake" else (1 - prob)

        logger.info(
            f"Инференс завершён: label={label}, prob={prob:.4f}, confidence={confidence:.4f}"
        )

        return PredictionResult(label, prob, confidence)
