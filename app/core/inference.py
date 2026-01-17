from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor

from app.services.logger import logger
from app.config.settings import DEVICE, DTYPE, CKPT_DIR, BASE_MODEL_ID
from app.core.model import DeepfakeSigLIP
from app.core.model_loader import load_weights_from_checkpoint
from app.core.preprocess import normalize_image_to_rgb
from app.core.video import VideoMeta, read_video_uniform_frames


@dataclass
class PredictionResult:
    label: str
    prob_deepfake: float
    confidence: float


@dataclass
class VideoPredictionResult(PredictionResult):
    per_frame_probs: List[float]
    meta: VideoMeta
    agg_method: str


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
        probs = self.predict_batch([image])
        prob = probs[0]

        label = "deepfake" if prob >= threshold else "real"
        confidence = prob if label == "deepfake" else (1 - prob)
        return PredictionResult(label, prob, confidence)

    @torch.no_grad()
    def predict_batch(self, images: List[PILImage.Image], batch_size: int = 16) -> List[float]:
        """
        Батч-инференс, значительно быстрее для видео.
        """
        if not images:
            return []

        probs: List[float] = []
        self.model.eval()

        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            chunk = [normalize_image_to_rgb(im) for im in chunk]
            inputs = self.processor(images=chunk, return_tensors="pt")

            pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float16)

            logits = self.model(pixel_values)  # [B, 1]
            p = torch.sigmoid(logits.float()).squeeze(-1)  # [B]
            probs.extend([float(x) for x in p.detach().cpu().tolist()])

        return probs

    @torch.no_grad()
    def predict_video(
        self,
        video_path,
        threshold: float = 0.5,
        agg_method: str = "median_of_means",
        chunk_count: int = 8,
        batch_size: int = 16,
        max_side: int = 768,
    ) -> VideoPredictionResult:
        frames, meta = read_video_uniform_frames(video_path, max_side=max_side)

        per_frame_probs = self.predict_batch(frames, batch_size=batch_size)

        prob = _aggregate_probs(per_frame_probs, method=agg_method, chunk_count=chunk_count)
        label = "deepfake" if prob >= threshold else "real"
        confidence = prob if label == "deepfake" else (1.0 - prob)

        return VideoPredictionResult(
            label=label,
            prob_deepfake=float(prob),
            confidence=float(confidence),
            per_frame_probs=per_frame_probs,
            meta=meta,
            agg_method=agg_method,
        )


def _aggregate_probs(probs: List[float], method: str, chunk_count: int = 8) -> float:
    if not probs:
        return 0.0

    method = method.lower().strip()

    if method == "mean":
        return float(sum(probs) / len(probs))

    if method == "trimmed_mean":
        s = sorted(probs)
        k = max(1, int(0.1 * len(s)))
        s2 = s[k:-k] if len(s) > 2 * k else s
        return float(sum(s2) / len(s2))

    k = max(1, min(chunk_count, len(probs)))
    chunks = []
    n = len(probs)
    for i in range(k):
        a = int(i * n / k)
        b = int((i + 1) * n / k)
        if a < b:
            chunks.append(probs[a:b])

    means = [sum(c) / len(c) for c in chunks if c]
    means.sort()
    mid = len(means) // 2
    return float(means[mid]) if len(means) % 2 == 1 else float(0.5 * (means[mid - 1] + means[mid]))
