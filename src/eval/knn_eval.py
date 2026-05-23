from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier

from src.features.extract_features import extract_split_features
from src.utils.metrics import (
    append_summary,
    save_classification_table,
    save_confusion_matrix,
    top1_accuracy,
)
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
        seed=args.seed,
    )


def run_knn(args) -> dict[int, float]:
    set_seed(args.seed)
    train = load_or_extract(args, "train")
    test = load_or_extract(args, "test")
    x_train = train["features"].float().numpy()
    y_train = train["labels"].numpy()
    x_test = test["features"].float().numpy()
    y_test = test["labels"].numpy()
    classes = list(test.get("classes") or train.get("classes") or [str(i) for i in sorted(set(y_train))])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    rows = []
    for k in args.k_values:
        effective_k = min(k, len(y_train))
        clf = KNeighborsClassifier(
            n_neighbors=effective_k,
            metric=args.metric,
            algorithm="brute",
            n_jobs=args.n_jobs,
        )
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = top1_accuracy(y_test, y_pred)
        results[k] = acc
        rows.append({"k": k, "effective_k": effective_k, "metric": args.metric, "top1_accuracy": acc})
        print(f"k-NN k={k} ({args.metric}) top-1 accuracy: {acc:.4f}")

        if k == args.k_values[0] or k == 10:
            save_confusion_matrix(y_test, y_pred, classes, output_dir / f"knn_k{k}_confusion_matrix.csv")
            save_classification_table(test["paths"], y_test, y_pred, classes, output_dir / f"knn_k{k}_report.csv")

        append_summary(
            {
                "scenario": "knn_zero_shot_transfer",
                "dataset": args.dataset,
                "method": f"sklearn_knn_{args.metric}",
                "backbone": args.model_name,
                "accuracy": acc,
                "top1": acc,
                "k": k,
                "feature_dim": int(x_train.shape[1]),
            },
            args.summary_csv,
        )

    pd.DataFrame(rows).to_csv(output_dir / "knn_metrics.csv", index=False)
    with (output_dir / "knn_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results, "feature_dim": int(x_train.shape[1])}, f, indent=2)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate sklearn k-NN on frozen DINOv2 features.")
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="outputs/knn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k_values", nargs="+", type=int, default=None)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "manhattan"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--summary_csv", default="results/summary.csv")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.k_values is None:
        args.k_values = [args.k]
    run_knn(args)


if __name__ == "__main__":
    main()
