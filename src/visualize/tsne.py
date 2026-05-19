from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw t-SNE for frozen DINOv2 features.")
    parser.add_argument("--features", required=True, help="Path to *_cls_features.pt.")
    parser.add_argument("--output", default="outputs/tsne.png")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    labels = payload["labels"].numpy()
    if features.shape[0] > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(features.shape[0], size=args.max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    features = StandardScaler().fit_transform(features)
    perplexity = min(args.perplexity, max(5, (features.shape[0] - 1) / 3))
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=args.seed,
    ).fit_transform(features)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        s=8,
        cmap="tab20",
        alpha=0.85,
        linewidths=0,
    )
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.title("DINOv2 CLS Feature t-SNE")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved t-SNE to {output}")


if __name__ == "__main__":
    main()
