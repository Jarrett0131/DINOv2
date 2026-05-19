from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

from src.data.datasets import load_image
from src.models.dinov2_model import load_dinov2


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_patch_pca(model, tensor: torch.Tensor, image: Image.Image, output: str | Path) -> None:
    result = model(tensor, return_patch_tokens=True)
    if result.patch is None:
        raise RuntimeError("Patch tokens are unavailable for this model.")
    patch_tokens = result.patch.squeeze(0).cpu().numpy()
    h = w = int(np.sqrt(patch_tokens.shape[0]))
    pca = PCA(n_components=1)
    patch_map = pca.fit_transform(patch_tokens).reshape(h, w)
    patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)
    patch_map = torch.from_numpy(patch_map)[None, None].float()
    patch_map = F.interpolate(
        patch_map,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()
    overlay_heatmap(image, patch_map, output)


def save_cls_patch_similarity(model, tensor: torch.Tensor, image: Image.Image, output: str | Path) -> None:
    result = model(tensor, return_patch_tokens=True)
    if result.patch is None:
        raise RuntimeError("Patch tokens are unavailable for this model.")
    cls = F.normalize(result.cls.squeeze(0), dim=0)
    patches = F.normalize(result.patch.squeeze(0), dim=1)
    scores = patches @ cls
    h = w = int(np.sqrt(scores.numel()))
    heatmap = scores.reshape(1, 1, h, w)
    heatmap = F.interpolate(
        heatmap,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    heatmap = heatmap.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    overlay_heatmap(image, heatmap, output)


def save_attention(model, tensor: torch.Tensor, image: Image.Image, output: str | Path) -> None:
    try:
        attn = model.get_last_selfattention(tensor)
    except NotImplementedError:
        save_cls_patch_similarity(model, tensor, image, output)
        return

    attn = attn.squeeze(0)
    cls_attention = attn[:, 0, 1:].mean(dim=0)
    h = w = int(np.sqrt(cls_attention.numel()))
    heatmap = cls_attention.reshape(1, 1, h, w)
    heatmap = F.interpolate(
        heatmap,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    heatmap = heatmap.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    overlay_heatmap(image, heatmap, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DINOv2 attention or patch PCA.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--output", default="outputs/attention_map.png")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--mode", choices=["attention", "patch_pca"], default="attention")
    args = parser.parse_args()

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    tensor, pil_image = load_image(args.image, image_size=args.image_size)
    tensor = tensor.to(device)
    model = load_dinov2(model_name=args.model_name, device=device)

    if args.mode == "patch_pca":
        save_patch_pca(model, tensor, pil_image, args.output)
    else:
        save_attention(model, tensor, pil_image, args.output)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
