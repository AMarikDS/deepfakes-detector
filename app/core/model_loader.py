"""
Загрузка весов модели из checkpoint.
Максимально близко к исходной логике backend.py.
"""

import os
import torch

from app.services.logger import logger


def load_weights_from_checkpoint(model, ckpt_dir: str):
    logger.info("Поиск весов модели...")
    paths = [
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]

    for fname in paths:
        full = os.path.join(ckpt_dir, fname)
        if os.path.exists(full):
            logger.info(f"Файл найден: {full}")

            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file

                state = load_file(full)
            else:
                state = torch.load(full, map_location="cpu")

            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info(
                f"Веса загружены. MISSING={len(missing)}, UNEXPECTED={len(unexpected)}"
            )
            return

    logger.error("ОШИБКА: файл весов не найден в директории.")
