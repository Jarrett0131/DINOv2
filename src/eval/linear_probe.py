from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    )


def run_linear_probe(args) -> float:
    train = load_or_extract(args, "train")
    test = load_or_extract(args, "test")
    x_train = train["features"].numpy()
    y_train = train["labels"].numpy()
    x_test = test["features"].numpy()
    y_test = test["labels"].numpy()

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            solver=args.solver,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        ),
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = top1_accuracy(y_test, y_pred)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = test.get("classes") or train.get("classes") or [str(i) for i in sorted(set(y_train))]
    save_confusion_matrix(
        y_test,
        y_pred,
        list(classes),
        output_dir / "linear_probe_confusion_matrix.csv",
    )
    save_classification_table(
        test["paths"],
        y_test,
        y_pred,
        list(classes),
        output_dir / "linear_probe_report.csv",
    )
    pd.DataFrame({"metric": ["top1_accuracy"], "value": [acc]}).to_csv(
        output_dir / "linear_probe_metrics.csv", index=False
    )
    torch.save(clf, output_dir / "linear_probe_classifier.pt")

    append_summary(
        {
            "scenario": "fine_grained_classification",
            "dataset": args.dataset,
            "method": "linear_probe",
            "backbone": args.model_name,
            "accuracy": acc,
            "top1": acc,
            "k": "",
            "feature_dim": int(x_train.shape[1]),
        },
        args.summary_csv,
    )
    print(f"Linear probe top-1 accuracy: {acc:.4f}")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear probe on frozen DINOv2 features.")
    parser.add_argument("--dataset", default="imagefolder")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs/linear_probe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--summary_csv", default="results/summary.csv")
    args = parser.parse_args()
    set_seed(args.seed)
    run_linear_probe(args)


if __name__ == "__main__":
    main()
