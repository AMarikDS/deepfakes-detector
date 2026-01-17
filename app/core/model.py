import torch
import torch.nn as nn
from transformers import AutoModel

from app.services.logger import logger


class DeepfakeSigLIP(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()

        logger.info(f"Загрузка backbone из {model_dir} (только локальные файлы)...")
        self.backbone = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True
        )
        logger.info("Backbone успешно загружен.")

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

        feat_dim = getattr(
            getattr(self.backbone.config, "vision_config", self.backbone.config),
            "hidden_size"
        )
        logger.info(f"Размер признаков backbone: {feat_dim}")

        self.norm = nn.LayerNorm(feat_dim)
        self.classifier = nn.Linear(feat_dim, 1)

    @torch.no_grad()
    def forward(self, pixel_values):
        out = self.backbone.vision_model(pixel_values=pixel_values)

        if getattr(out, "image_embeds", None) is not None:
            feats = out.image_embeds
        elif getattr(out, "pooler_output", None) is not None:
            feats = out.pooler_output
        else:
            feats = out.last_hidden_state.mean(dim=1)

        feats = self.norm(feats)
        return self.classifier(feats)
