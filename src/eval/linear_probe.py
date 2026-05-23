from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data.datasets import build_loader
from src.models.dinov2_model import load_dinov2
from src.utils.metrics import (
    append_summary,
    save_classification_table,
    save_confusion_matrix,
    top1_accuracy,
)
from src.utils.seed import set_seed


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


@torch.no_grad()
def infer_feature_dim(backbone: nn.Module, loader, device: str) -> int:
    images, _, _ = next(iter(loader))
    features = backbone(images.to(device, non_blocking=True)).cls
    if features.ndim != 2:
        raise RuntimeError(f"Expected CLS features with shape [B, D], got {tuple(features.shape)}.")
    return int(features.shape[1])


def build_loaders(args):
    train_loader = build_loader(
        dataset=args.dataset,
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        shuffle=True,
        seed=args.seed,
    )
    test_loader = build_loader(
        dataset=args.dataset,
        data_root=args.data_root,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        shuffle=False,
        seed=args.seed,
    )
    return train_loader, test_loader


def train_one_epoch(
    backbone: nn.Module,
    classifier: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    max_batches: int | None = None,
) -> tuple[float, float]:
    backbone.eval()
    classifier.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=f"linear train epoch {epoch}", leave=False)

    for batch_idx, (images, labels, _) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            features = backbone(images).cls
        logits = classifier(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size
        progress.set_postfix(loss=total_loss / max(total, 1), acc=correct / max(total, 1))

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    backbone: nn.Module,
    classifier: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    max_batches: int | None = None,
) -> tuple[float, float, list[int], list[int], list[str]]:
    backbone.eval()
    classifier.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []

    for batch_idx, (images, labels, batch_paths) in enumerate(tqdm(loader, desc="linear eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = classifier(backbone(images).cls)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((preds == labels).sum().item())
        total += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        paths.extend(list(batch_paths))

    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred, paths


def save_checkpoint(
    path: Path,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    feature_dim: int,
    classes: list[str],
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "feature_dim": feature_dim,
            "num_classes": len(classes),
            "classes": classes,
            "model_name": args.model_name,
            "image_size": args.image_size,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str | Path, classifier: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_acc", 0.0))


def run_linear_probe(args) -> float:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoints" / "best_linear_probe.pt"

    train_loader, test_loader = build_loaders(args)
    classes = list(getattr(train_loader.dataset, "classes", []))
    if not classes:
        raise RuntimeError("Dataset must expose class names for linear probe evaluation.")

    backbone = load_dinov2(args.model_name, device=args.device, source=args.model_source)
    feature_dim = infer_feature_dim(backbone, train_loader, args.device)
    classifier = LinearClassifier(feature_dim, len(classes)).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, classifier, optimizer, args.device)

    rows = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            backbone,
            classifier,
            train_loader,
            criterion,
            optimizer,
            args.device,
            epoch,
            args.max_train_batches,
        )
        test_loss, test_acc, _, _, _ = evaluate(
            backbone,
            classifier,
            test_loader,
            criterion,
            args.device,
            args.max_eval_batches,
        )
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )
        pd.DataFrame(rows).to_csv(output_dir / "linear_probe_metrics.csv", index=False)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f}"
        )

        if test_acc >= best_acc:
            best_acc = test_acc
            save_checkpoint(
                checkpoint_path,
                classifier,
                optimizer,
                epoch,
                best_acc,
                feature_dim,
                classes,
                args,
            )

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        classifier.load_state_dict(checkpoint["classifier_state_dict"])
        best_acc = float(checkpoint["best_acc"])

    _, final_acc, y_true, y_pred, paths = evaluate(
        backbone,
        classifier,
        test_loader,
        criterion,
        args.device,
        args.max_eval_batches,
    )
    y_true_np = torch.tensor(y_true).numpy()
    y_pred_np = torch.tensor(y_pred).numpy()
    save_confusion_matrix(y_true_np, y_pred_np, classes, output_dir / "linear_probe_confusion_matrix.csv")
    save_classification_table(paths, y_true_np, y_pred_np, classes, output_dir / "linear_probe_report.csv")
    with (output_dir / "linear_probe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "top1_accuracy": top1_accuracy(y_true_np, y_pred_np),
                "best_top1_accuracy": best_acc,
                "final_eval_accuracy": final_acc,
                "feature_dim": feature_dim,
                "num_classes": len(classes),
                "checkpoint": str(checkpoint_path),
            },
            f,
            indent=2,
        )

    append_summary(
        {
            "scenario": "fine_grained_classification",
            "dataset": args.dataset,
            "method": "linear_probe",
            "backbone": args.model_name,
            "accuracy": best_acc,
            "top1": best_acc,
            "k": "",
            "feature_dim": feature_dim,
        },
        args.summary_csv,
    )
    print(f"Best linear probe top-1 accuracy: {best_acc:.4f}")
    return best_acc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a linear classifier on frozen DINOv2 CLS features.")
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--model_source", default="torchhub")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--output_dir", default="outputs/linear_probe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    parser.add_argument("--summary_csv", default="results/summary.csv")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_linear_probe(args)


if __name__ == "__main__":
    main()
