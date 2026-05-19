from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.datasets import build_loader
from src.models.dinov2_model import load_dinov2
from src.utils.seed import set_seed


def extract_split_features(
    dataset: str,
    data_root: str,
    split: str,
    model_name: str,
    batch_size: int,
    output_dir: str | Path,
    device: str,
    image_size: int = 518,
    num_workers: int = 4,
    save_patch_tokens: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = build_loader(
        dataset=dataset,
        data_root=data_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        shuffle=False,
    )
    model = load_dinov2(model_name=model_name, device=device)

    cls_features = []
    patch_features = []
    labels = []
    paths = []

    for images, batch_labels, batch_paths in tqdm(loader, desc=f"extract {split}"):
        images = images.to(device, non_blocking=True)
        output = model(images, return_patch_tokens=save_patch_tokens)
        cls_features.append(output.cls.cpu())
        if save_patch_tokens and output.patch is not None:
            patch_features.append(output.patch.cpu())
        labels.append(batch_labels.cpu())
        paths.extend(list(batch_paths))

    payload = {
        "features": torch.cat(cls_features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "paths": paths,
        "classes": getattr(loader.dataset, "classes", None),
        "model_name": model_name,
        "dataset": dataset,
        "split": split,
    }
    torch.save(payload, output_dir / f"{split}_cls_features.pt")

    if save_patch_tokens and patch_features:
        patch_payload = {
            "patch_tokens": torch.cat(patch_features, dim=0),
            "labels": payload["labels"],
            "paths": paths,
            "model_name": model_name,
            "dataset": dataset,
            "split": split,
        }
        torch.save(patch_payload, output_dir / f"{split}_patch_tokens.pt")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen DINOv2 CLS features.")
    parser.add_argument("--dataset", default="imagefolder")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs/dinov2_vitb14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--save_patch_tokens", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    for split in args.splits:
        extract_split_features(
            dataset=args.dataset,
            data_root=args.data_root,
            split=split,
            model_name=args.model_name,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
            num_workers=args.num_workers,
            save_patch_tokens=args.save_patch_tokens,
        )


if __name__ == "__main__":
    main()
