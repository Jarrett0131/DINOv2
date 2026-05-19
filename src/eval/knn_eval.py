from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.features.extract_features import extract_split_features
from src.utils.metrics import append_summary, top1_accuracy
from src.utils.seed import set_seed


def load_or_extract(args, split: str) -> dict:
    path = Path(args.output_dir) / f"{split}_cls_features.pt"
    if path.exists() and not args.force_extract:
        return torch.load(path, map_location="cpu", weights_only=False)
    return extract_split_features(
        dataset=args.dataset,
        data_root=args.data_root,
        split=split,
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        image_size=args.image_size,
        num_workers=args.num_workers,
        save_patch_tokens=False,
    )


def cosine_knn(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    k: int,
    batch_size: int = 1024,
) -> np.ndarray:
    train_features = F.normalize(train_features.float(), dim=1)
    test_features = F.normalize(test_features.float(), dim=1)
    predictions = []
    classes = torch.unique(train_labels).sort().values

    for start in range(0, test_features.shape[0], batch_size):
        sims = test_features[start : start + batch_size] @ train_features.T
        topk = sims.topk(k=min(k, train_features.shape[0]), dim=1).indices
        topk_labels = train_labels[topk]
        votes = torch.zeros(topk_labels.shape[0], int(classes.max().item()) + 1)
        votes.scatter_add_(1, topk_labels, torch.ones_like(topk_labels, dtype=votes.dtype))
        predictions.append(votes.argmax(dim=1))
    return torch.cat(predictions).cpu().numpy()


def run_knn(args) -> dict[int, float]:
    train = load_or_extract(args, "train")
    test = load_or_extract(args, "test")
    y_test = test["labels"].numpy()
    results = {}
    rows = []

    for k in args.k_values:
        y_pred = cosine_knn(
            train["features"],
            train["labels"].long(),
            test["features"],
            k=k,
            batch_size=args.knn_batch_size,
        )
        acc = top1_accuracy(y_test, y_pred)
        results[k] = acc
        rows.append({"k": k, "top1_accuracy": acc})
        append_summary(
            {
                "scenario": "knn_zero_shot_transfer",
                "dataset": args.dataset,
                "method": "cosine_knn",
                "backbone": args.model_name,
                "accuracy": acc,
                "top1": acc,
                "k": k,
                "feature_dim": int(train["features"].shape[1]),
            },
            args.summary_csv,
        )
        print(f"k-NN k={k} top-1 accuracy: {acc:.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.output_dir) / "knn_metrics.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cosine k-NN on frozen DINOv2 features.")
    parser.add_argument("--dataset", default="imagefolder")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs/knn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--knn_batch_size", type=int, default=1024)
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--summary_csv", default="results/summary.csv")
    args = parser.parse_args()
    set_seed(args.seed)
    run_knn(args)


if __name__ == "__main__":
    main()
