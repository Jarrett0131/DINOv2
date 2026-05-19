from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DINOv2Output:
    cls: torch.Tensor
    patch: torch.Tensor | None = None
    attentions: torch.Tensor | None = None


class DINOv2FeatureExtractor(nn.Module):
    """Frozen DINOv2 wrapper that returns CLS and optional patch tokens."""

    def __init__(self, model_name: str = "dinov2_vitb14", source: str = "torchhub") -> None:
        super().__init__()
        if source != "torchhub":
            raise ValueError("Only source='torchhub' is implemented for DINOv2 loading.")
        self.model_name = model_name
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor, return_patch_tokens: bool = False) -> DINOv2Output:
        outputs = self.backbone.forward_features(images)
        cls = outputs["x_norm_clstoken"]
        patch = outputs.get("x_norm_patchtokens") if return_patch_tokens else None
        return DINOv2Output(cls=cls, patch=patch)

    @torch.no_grad()
    def get_last_selfattention(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "get_last_selfattention"):
            return self.backbone.get_last_selfattention(images)
        raise NotImplementedError(
            f"{self.model_name} from torch.hub does not expose get_last_selfattention."
        )


def load_dinov2(
    model_name: str = "dinov2_vitb14",
    device: str = "cuda",
    source: str = "torchhub",
) -> DINOv2FeatureExtractor:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = DINOv2FeatureExtractor(model_name=model_name, source=source)
    return model.to(device).eval()
