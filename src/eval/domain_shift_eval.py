from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.knn_eval import cosine_knn
from src.features.extract_features import extract_split_features
from src.utils.metrics import append_summary, top1_accuracy
from src.utils.seed import set_seed


def feature_path(output_dir: str | Path, prefix: str, split: str) -> Path:
    return Path(output_dir) / f"{prefix}_{split}_cls_features.pt"


def load_or_extract_domain(args, root: str, split: str, prefix: str) -> dict:
    path = feature_path(args.output_dir, prefix, split)
    if path.exists() and not args.force_extract:
        return torch.load(path, map_location="cpu", weights_only=False)

    payload = extract_split_features(
        dataset=args.dataset,
        data_root=root,
        split=split,
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        image_size=args.image_size,
        num_workers=args.num_workers,
        save_patch_tokens=False,
    )
    default_path = Path(args.output_dir) / f"{split}_cls_features.pt"
    if default_path.exists():
        default_path.replace(path)
    return payload


def evaluate_linear(args, source_train: dict, source_test: dict, target_test: dict) -> tuple[float, float]:
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
    clf.fit(source_train["features"].numpy(), source_train["labels"].numpy())
    source_pred = clf.predict(source_test["features"].numpy())
    target_pred = clf.predict(target_test["features"].numpy())
    source_acc = top1_accuracy(source_test["labels"].numpy(), source_pred)
    target_acc = top1_accuracy(target_test["labels"].numpy(), target_pred)
    torch.save(clf, Path(args.output_dir) / "domain_shift_linear_probe.pt")
    return source_acc, target_acc


def evaluate_knn(args, source_train: dict, source_test: dict, target_test: dict) -> tuple[float, float]:
    source_pred = cosine_knn(
        source_train["features"],
        source_train["labels"].long(),
        source_test["features"],
        k=args.k,
        batch_size=args.knn_batch_size,
    )
    target_pred = cosine_knn(
        source_train["features"],
        source_train["labels"].long(),
        target_test["features"],
        k=args.k,
        batch_size=args.knn_batch_size,
    )
    source_acc = top1_accuracy(source_test["labels"].numpy(), source_pred)
    target_acc = top1_accuracy(target_test["labels"].numpy(), target_pred)
    return source_acc, target_acc


def run_domain_shift(args) -> dict:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    source_train = load_or_extract_domain(args, args.source_root, "train", "source")
    source_test = load_or_extract_domain(args, args.source_root, "test", "source")
    target_test = load_or_extract_domain(args, args.target_root, "test", "target")

    if args.method == "linear":
        source_acc, target_acc = evaluate_linear(args, source_train, source_test, target_test)
        method_name = "linear_probe_source_to_target"
        k_value = ""
    else:
        source_acc, target_acc = evaluate_knn(args, source_train, source_test, target_test)
        method_name = "cosine_knn_source_to_target"
        k_value = args.k

    drop = source_acc - target_acc
    row = {
        "method": method_name,
        "source_accuracy": source_acc,
        "target_accuracy": target_acc,
        "accuracy_drop": drop,
        "k": k_value,
        "feature_dim": int(source_train["features"].shape[1]),
    }
    pd.DataFrame([row]).to_csv(Path(args.output_dir) / "domain_shift_metrics.csv", index=False)

    append_summary(
        {
            "scenario": "domain_shift_robustness",
            "dataset": args.dataset,
            "method": method_name,
            "backbone": args.model_name,
            "accuracy": target_acc,
            "top1": target_acc,
            "k": k_value,
            "feature_dim": int(source_train["features"].shape[1]),
        },
        args.summary_csv,
    )
    print(
        "Domain shift: "
        f"source_acc={source_acc:.4f}, target_acc={target_acc:.4f}, drop={drop:.4f}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen DINOv2 under domain shift.")
    parser.add_argument("--dataset", default="imagefolder")
    parser.add_argument("--data_root", default=None, help="Alias for source_root for script compatibility.")
    parser.add_argument("--source_root", default=None)
    parser.add_argument("--target_root", required=True)
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs/domain_shift")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["linear", "knn"], default="linear")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--knn_batch_size", type=int, default=1024)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--summary_csv", default="results/summary.csv")
    args = parser.parse_args()
    args.source_root = args.source_root or args.data_root
    if args.source_root is None:
        raise ValueError("Provide --source_root or --data_root.")
    set_seed(args.seed)
    run_domain_shift(args)


if __name__ == "__main__":
    main()
