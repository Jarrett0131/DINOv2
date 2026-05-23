from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.eval.knn_eval import run_knn
from src.eval.linear_probe import run_linear_probe
from src.features.extract_features import extract_split_features
from src.utils.seed import set_seed


def run_tsne(feature_path: Path, output_path: Path, max_samples: int, seed: int, perplexity: float) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.visualize.tsne",
        "--features",
        str(feature_path),
        "--output",
        str(output_path),
        "--max_samples",
        str(max_samples),
        "--seed",
        str(seed),
        "--perplexity",
        str(perplexity),
    ]
    subprocess.run(cmd, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified DINOv2 frozen-feature experiment runner.")
    parser.add_argument("--mode", choices=["all", "extract", "linear", "knn", "tsne"], default="all")
    parser.add_argument("--dataset", default="auto", help="auto, cub, flowers, cars, or imagefolder")
    parser.add_argument("--data_root", default="src/data/CUB_200_2011")
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--model_source", default="torchhub")
    parser.add_argument("--output_dir", default="outputs/baseline")
    parser.add_argument("--summary_csv", default="results/summary.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "manhattan"])
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--tsne_max_samples", type=int, default=2000)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--max_train_batches", type=int, default=None, help="Debug only: limit linear-train batches.")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Debug only: limit eval batches.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"all", "extract"}:
        for split in ("train", "test"):
            extract_split_features(
                dataset=args.dataset,
                data_root=args.data_root,
                split=split,
                model_name=args.model_name,
                batch_size=args.batch_size,
                output_dir=output_dir,
                device=args.device,
                image_size=args.image_size,
                num_workers=args.num_workers,
                save_patch_tokens=False,
                seed=args.seed,
            )

    if args.mode in {"all", "linear"}:
        run_linear_probe(args)

    if args.mode in {"all", "knn"}:
        args.k_values = [args.k]
        run_knn(args)

    if args.mode in {"all", "tsne"}:
        feature_path = output_dir / "test_cls_features.pt"
        if not feature_path.exists() or args.force_extract:
            extract_split_features(
                dataset=args.dataset,
                data_root=args.data_root,
                split="test",
                model_name=args.model_name,
                batch_size=args.batch_size,
                output_dir=output_dir,
                device=args.device,
                image_size=args.image_size,
                num_workers=args.num_workers,
                save_patch_tokens=False,
                seed=args.seed,
            )
        run_tsne(
            feature_path,
            output_dir / "tsne" / "test_tsne.png",
            args.tsne_max_samples,
            args.seed,
            args.tsne_perplexity,
        )


if __name__ == "__main__":
    main()
