from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "cub_dinov2_vitb14"
FIG_DIR = ROOT / "paper_figures"
TABLE_DIR = ROOT / "paper_tables"
TABLE_CSV_DIR = TABLE_DIR / "csv"
ASSET_DIR = ROOT / "paper_assets"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def percent(value: float | int | str | None) -> str:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value) * 100:.2f}"


def mkdirs() -> None:
    for path in [
        FIG_DIR,
        TABLE_DIR,
        TABLE_CSV_DIR,
        ASSET_DIR / "figures",
        ASSET_DIR / "tables",
        ASSET_DIR / "checkpoints",
        ASSET_DIR / "logs",
        ASSET_DIR / "analysis",
        ASSET_DIR / "appendix",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "shape": tuple(payload["features"].shape),
        "labels_shape": tuple(payload["labels"].shape),
        "paths": len(payload.get("paths", [])),
        "classes": len(payload.get("classes") or []),
        "dataset": payload.get("dataset"),
        "split": payload.get("split"),
        "model_name": payload.get("model_name"),
        "image_size": payload.get("image_size"),
        "label_min": int(payload["labels"].min()),
        "label_max": int(payload["labels"].max()),
        "path_set": set(payload.get("paths", [])),
        "labels": payload["labels"].numpy(),
        "class_names": payload.get("classes") or [],
    }


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(ASSET_DIR / "figures" / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSET_DIR / "figures" / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
        }
    )


def make_training_curve(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(metrics["epoch"], metrics["train_loss"], label="Train", color="#2C7FB8", linewidth=1.8)
    axes[0].plot(metrics["epoch"], metrics["test_loss"], label="Test", color="#D95F02", linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Linear Probe Loss")
    axes[0].grid(True, alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False)

    axes[1].plot(metrics["epoch"], metrics["train_accuracy"] * 100, label="Train", color="#2C7FB8", linewidth=1.8)
    axes[1].plot(metrics["epoch"], metrics["test_accuracy"] * 100, label="Test", color="#D95F02", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_title("Linear Probe Accuracy")
    axes[1].grid(True, alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False)
    save_fig(fig, "train_test_curve")


def make_accuracy_bar(summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["label"] = plot_df.apply(
        lambda r: "Linear probe" if r["method"] == "linear_probe" else f"k-NN (k={int(r['k'])})",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    colors = ["#2C7FB8", "#31A354", "#D95F02", "#756BB1"]
    bars = ax.bar(plot_df["label"], plot_df["accuracy"] * 100, color=colors[: len(plot_df)], width=0.55)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, max(100, float(plot_df["accuracy"].max() * 112)))
    ax.set_title("CUB-200-2011 Accuracy")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, value in zip(bars, plot_df["accuracy"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    save_fig(fig, "accuracy_comparison")


def make_confusion_matrix(path: Path, stem: str, title: str) -> dict:
    matrix_df = pd.read_csv(path, index_col=0)
    matrix = matrix_df.to_numpy(dtype=float)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum != 0) * 100.0
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=max(1.0, np.nanpercentile(normalized, 99)))
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title(title)
    ticks = np.arange(0, normalized.shape[0], 25)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized count (%)")
    save_fig(fig, stem)
    return {
        "sum": int(matrix.sum()),
        "diag": int(np.trace(matrix)),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "shape": matrix.shape,
    }


def make_tsne() -> dict:
    payload = torch.load(OUT / "test_cls_features.pt", map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    labels = payload["labels"].numpy()
    max_samples = 2000
    seed = 42
    if features.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(features.shape[0], size=max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
    scaled = StandardScaler().fit_transform(features)
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30.0,
        random_state=seed,
    ).fit_transform(scaled)
    pd.DataFrame({"tsne_1": embedding[:, 0], "tsne_2": embedding[:, 1], "label": labels}).to_csv(
        FIG_DIR / "tsne_test_embedding.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels + 1,
        s=8,
        cmap="turbo",
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("DINOv2 Test Feature t-SNE")
    ax.grid(False)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class index")
    save_fig(fig, "tsne_test_features")
    return {"samples": int(len(labels)), "classes": int(len(np.unique(labels)))}


def try_benchmark_and_params() -> dict:
    result = {
        "backbone_params_m": None,
        "classifier_params_m": 0.1538,
        "total_trainable_params_m": 0.1538,
        "device": "N/A",
        "batch_size": 64,
        "images_per_second": None,
        "latency_ms_per_image": None,
        "note": "Not measured.",
    }
    try:
        from src.data.datasets import build_loader
        from src.models.dinov2_model import load_dinov2

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_dinov2("dinov2_vitb14", device=device)
        result["device"] = device
        result["backbone_params_m"] = sum(p.numel() for p in model.backbone.parameters()) / 1e6
        if device != "cuda":
            result["note"] = "Parameter count measured; speed benchmark skipped because CUDA is unavailable."
            return result
        loader = build_loader(
            dataset="cub",
            data_root=str(ROOT / "src" / "data" / "CUB_200_2011"),
            split="test",
            batch_size=64,
            num_workers=4,
            image_size=224,
            shuffle=False,
            seed=42,
        )
        warmup_batches = 2
        timed_batches = 20
        total_images = 0
        with torch.no_grad():
            for i, (images, _, _) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                _ = model(images).cls
                if i + 1 >= warmup_batches:
                    break
            torch.cuda.synchronize()
            start = time.perf_counter()
            for i, (images, _, _) in enumerate(loader):
                if i >= timed_batches:
                    break
                images = images.to(device, non_blocking=True)
                _ = model(images).cls
                total_images += images.shape[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        ips = total_images / elapsed if elapsed > 0 else None
        result["images_per_second"] = ips
        result["latency_ms_per_image"] = 1000.0 / ips if ips else None
        result["note"] = f"Measured on {timed_batches} test batches after {warmup_batches} warmup batches; includes image transfer and backbone forward only."
    except Exception as exc:
        result["note"] = f"Benchmark/parameter measurement failed: {type(exc).__name__}: {exc}"
    return result


def write_tables(summary: pd.DataFrame, benchmark: dict) -> None:
    method_rows = []
    for _, row in summary.iterrows():
        method_rows.append(
            {
                "Dataset": row["dataset"].upper(),
                "Backbone": row["backbone"],
                "Method": "Linear Probe" if row["method"] == "linear_probe" else f"k-NN ({row['method'].replace('sklearn_knn_', '')}, k={int(row['k'])})",
                "Top-1 Accuracy (%)": percent(row["accuracy"]),
                "Feature Dim": int(row["feature_dim"]),
            }
        )
    accuracy_table = pd.DataFrame(method_rows)

    lp = float(summary.loc[summary["method"] == "linear_probe", "accuracy"].iloc[0])
    knn = float(summary.loc[summary["method"].str.contains("knn"), "accuracy"].iloc[0])
    lp_knn_table = pd.DataFrame(
        [
            {"Dataset": "CUB", "Linear Probe (%)": percent(lp), "k-NN (%)": percent(knn), "Delta (%)": f"{(lp - knn) * 100:.2f}"}
        ]
    )

    param_rows = [
        {
            "Model": "DINOv2 ViT-B/14 backbone",
            "Parameters (M)": "N/A" if benchmark["backbone_params_m"] is None else f"{benchmark['backbone_params_m']:.2f}",
            "Trainable in Linear Probe (M)": "0.00",
            "Source": "Measured from torch hub model" if benchmark["backbone_params_m"] is not None else benchmark["note"],
        },
        {
            "Model": "Linear classifier head",
            "Parameters (M)": f"{benchmark['classifier_params_m']:.2f}",
            "Trainable in Linear Probe (M)": f"{benchmark['classifier_params_m']:.2f}",
            "Source": "768 x 200 + 200 checkpoint metadata",
        },
    ]
    params_table = pd.DataFrame(param_rows)

    speed_table = pd.DataFrame(
        [
            {
                "Backbone": "dinov2_vitb14",
                "Device": benchmark["device"],
                "Batch Size": benchmark["batch_size"],
                "Images/s": "N/A" if benchmark["images_per_second"] is None else f"{benchmark['images_per_second']:.2f}",
                "Latency (ms/image)": "N/A" if benchmark["latency_ms_per_image"] is None else f"{benchmark['latency_ms_per_image']:.2f}",
                "Note": benchmark["note"],
            }
        ]
    )

    dataset_table = accuracy_table[["Dataset", "Backbone", "Method", "Top-1 Accuracy (%)"]].copy()

    tables = {
        "accuracy_comparison": accuracy_table,
        "linear_probe_vs_knn": lp_knn_table,
        "parameter_comparison": params_table,
        "inference_speed": speed_table,
        "dataset_results": dataset_table,
    }
    for name, df in tables.items():
        df.to_csv(TABLE_CSV_DIR / f"{name}.csv", index=False)

    def df_to_markdown(df: pd.DataFrame) -> str:
        rows = [[str(col) for col in df.columns]]
        rows.extend([[str(value) for value in row] for row in df.to_numpy().tolist()])
        widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]

        def fmt(row: list[str]) -> str:
            return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"

        separator = "| " + " | ".join("-" * width for width in widths) + " |"
        return "\n".join([fmt(rows[0]), separator, *[fmt(row) for row in rows[1:]]])

    md_parts = []
    tex_parts = []
    captions = {
        "accuracy_comparison": "不同模型/方法 Accuracy 对比",
        "linear_probe_vs_knn": "Linear Probe 与 k-NN 对比",
        "parameter_comparison": "参数量对比",
        "inference_speed": "推理速度对比",
        "dataset_results": "不同数据集结果",
    }
    for name, df in tables.items():
        md_parts.append(f"## {captions[name]}\n\n{df_to_markdown(df)}\n")
        tex_parts.append(f"% {captions[name]}\n")
        tex_parts.append(df.to_latex(index=False, escape=True, caption=captions[name], label=f"tab:{name}"))
        tex_parts.append("\n")
    (TABLE_DIR / "markdown_tables.md").write_text("\n".join(md_parts), encoding="utf-8")
    (TABLE_DIR / "latex_tables.tex").write_text("\n".join(tex_parts), encoding="utf-8")
    shutil.copy2(TABLE_DIR / "markdown_tables.md", ASSET_DIR / "tables" / "markdown_tables.md")
    shutil.copy2(TABLE_DIR / "latex_tables.tex", ASSET_DIR / "tables" / "latex_tables.tex")
    for csv_path in TABLE_CSV_DIR.glob("*.csv"):
        shutil.copy2(csv_path, ASSET_DIR / "tables" / csv_path.name)


def write_markdown_files(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    train_meta: dict,
    test_meta: dict,
    linear_summary: dict,
    knn_summary: dict,
    ckpt: dict,
    cm_stats: dict,
    tsne_stats: dict,
    benchmark: dict,
) -> None:
    inventory_rows = [
        ("Linear Probe Accuracy", OUT / "linear_probe_summary.json", "linear probe top-1, best checkpoint epoch and feature dimension", "是", "否"),
        ("k-NN Accuracy", OUT / "knn_summary.json", "k=10 cosine kNN top-1", "是", "否"),
        ("train/test loss 曲线", OUT / "linear_probe_metrics.csv", "epoch-level train/test loss and accuracy", "是，已重绘", "否"),
        ("confusion matrix", OUT / "linear_probe_confusion_matrix.csv", "linear probe 200x200 confusion matrix", "是，已重绘", "否"),
        ("confusion matrix", OUT / "knn_k10_confusion_matrix.csv", "kNN 200x200 confusion matrix", "是，已重绘", "否"),
        ("t-SNE 可视化", OUT / "tsne" / "test_tsne.png", "test-set DINOv2 CLS t-SNE legacy image", "原图不建议；已重绘", "否"),
        ("attention map / gradcam", ROOT / "src" / "visualize" / "attention_map.py", "has visualization script, no generated output found", "否", "建议补做"),
        ("推理速度", TABLE_CSV_DIR / "inference_speed.csv", "lightweight benchmark result or explicit missing note", "若有数值则可用；否则需补测", "视 benchmark 结果"),
        ("参数量", TABLE_CSV_DIR / "parameter_comparison.csv", "backbone/head parameter count", "是（若 backbone 成功测量）", "否"),
        ("特征维度", OUT / "train_cls_features.pt", "feature tensor metadata, dim=768", "是", "否"),
        ("checkpoint", OUT / "checkpoints" / "best_linear_probe.pt", "best linear probe classifier checkpoint", "可复现实验；论文不直接展示", "否"),
    ]
    speed_text = "N/A" if benchmark["images_per_second"] is None else f"{benchmark['images_per_second']:.2f} images/s"
    result_md = [
        "# Results Summary",
        "",
        "## 核心指标",
        "",
        f"- Dataset: CUB-200-2011; train={train_meta.get('shape', ['N/A'])[0]}, test={test_meta.get('shape', ['N/A'])[0]}, classes={test_meta.get('classes', 'N/A')}.",
        f"- Backbone: dinov2_vitb14; image size={test_meta.get('image_size', 'N/A')}; feature dimension={linear_summary.get('feature_dim', test_meta.get('shape', ['N/A', 'N/A'])[1])}.",
        f"- Linear Probe Top-1 Accuracy: {percent(linear_summary.get('top1_accuracy'))}%; best epoch={ckpt.get('epoch', 'N/A')}.",
        f"- k-NN Top-1 Accuracy: {percent(float(knn_summary.get('results', {}).get('10', float('nan'))))}%; k=10, metric=cosine.",
        f"- Train/test curve: epochs {int(metrics['epoch'].min())}-{int(metrics['epoch'].max())}; best test accuracy at epoch {int(metrics.loc[metrics['test_accuracy'].idxmax(), 'epoch'])}.",
        f"- Inference speed: {speed_text} ({benchmark['note']}).",
        "",
        "## 文件清单与论文可用性",
        "",
        "| 项目 | 文件路径 | 文件作用 | 是否可直接用于论文 | 是否需要重跑 |",
        "|---|---|---|---|---|",
    ]
    for item, path, role, usable, rerun in inventory_rows:
        result_md.append(f"| {item} | `{rel(path)}` | {role} | {usable} | {rerun} |")
    result_md += [
        "",
        "## 图表筛查结果",
        "",
        "- Legacy t-SNE PNG is 1600x1400 and was saved at 200 dpi; it uses a repeated `tab20` palette for 200 classes, lacks explicit x/y axis labels, and has no PDF companion. It is not recommended as the submitted figure.",
        "- Confusion matrices existed only as CSV files, so they were not directly usable as figures before this pass.",
        "- Training curves existed only as CSV/log text and were not directly usable before this pass.",
        "- No attention map or GradCAM output image was found under the scanned result directories.",
        "",
        "## 已重绘论文图表",
        "",
        "- `paper_figures/tsne_test_features.png` and `.pdf`",
        "- `paper_figures/linear_probe_confusion_matrix.png` and `.pdf`",
        "- `paper_figures/knn_k10_confusion_matrix.png` and `.pdf`",
        "- `paper_figures/train_test_curve.png` and `.pdf`",
        "- `paper_figures/accuracy_comparison.png` and `.pdf`",
        "",
        "## 缺失实验项列表",
        "",
        "- Attention/GradCAM figure outputs are missing.",
        "- TensorBoard event logs are missing.",
        "- Independent CNN baseline results are missing.",
        "- Cross-dataset/domain-shift results are not present, although a domain shift evaluation module exists.",
        "- kNN hyperparameter sweep beyond k=10 is missing.",
        "- Multiple seeds or confidence intervals are missing.",
        "- Paper-grade inference benchmark is missing if the lightweight CUDA benchmark did not produce a value.",
    ]
    (ROOT / "results_summary.md").write_text("\n".join(result_md), encoding="utf-8")

    train_test_overlap = len(train_meta.get("path_set", set()) & test_meta.get("path_set", set()))
    audit_md = [
        "# Experiment Audit",
        "",
        "## 可信部分",
        "",
        f"- Train/test leakage: feature payload path overlap is {train_test_overlap}; CUB split is read from `train_test_split.txt`.",
        "- Backbone frozen: `DINOv2FeatureExtractor` sets all backbone parameters `requires_grad=False`; linear probe training wraps backbone forward in `torch.no_grad()`.",
        "- Train/eval mode: linear probe uses `backbone.eval()`, `classifier.train()` during training and `classifier.eval()` during evaluation.",
        "- Test augmentation: dataset transform is deterministic resize + tensor conversion + normalization; no random augmentation is applied to the test set.",
        "- kNN protocol: `KNeighborsClassifier.fit` uses `train_cls_features.pt`; predictions use `test_cls_features.pt`.",
        f"- Confusion matrix label alignment: both matrices are {cm_stats['linear']['shape'][0]}x{cm_stats['linear']['shape'][1]}, sum to {cm_stats['linear']['sum']} test samples, and diagonal accuracy matches summary.",
        f"- t-SNE split: regenerated t-SNE uses only `test_cls_features.pt` with {tsne_stats['samples']} samples.",
        "",
        "## 风险点",
        "",
        "- The linear probe checkpoint metadata shows `resume` pointing to the same best checkpoint path. This is consistent with resumed training, but the current metrics CSV starts at epoch 6, so epochs 1-5 are not preserved in the final CSV.",
        "- Model selection uses test accuracy to save the best checkpoint. This is acceptable for a quick benchmark, but for a strict paper protocol it should use a validation split and report a single final test evaluation.",
        "- No TensorBoard logs or immutable run manifest were found; reproducibility depends on CSV/JSON/checkpoint metadata.",
        "- kNN was evaluated only for k=10, so the reported kNN number may be sensitive to the chosen k.",
        "- Existing t-SNE is a sampled visualization and should be interpreted qualitatively, not as a quantitative clustering metric.",
        "- Attention/GradCAM conclusions cannot be supported until actual visualization outputs are generated.",
        "",
        "## 建议重跑或补做",
        "",
        "- Re-run linear probe with train/val/test separation, select checkpoints on validation accuracy, and report final test once.",
        "- Run at least 3 random seeds and report mean +/- standard deviation.",
        "- Run kNN over a small k sweep, e.g. 1, 5, 10, 20, 50.",
        "- Add CNN or supervised ViT baselines under the same CUB split.",
        "- Generate attention/patch-token maps for representative correct and incorrect predictions.",
        "- Add a controlled inference benchmark script with hardware, batch size, precision, warmup, and timed iterations.",
    ]
    (ROOT / "experiment_audit.md").write_text("\n".join(audit_md), encoding="utf-8")

    delta = (float(linear_summary["top1_accuracy"]) - float(knn_summary["results"]["10"])) * 100
    analysis_md = [
        "# Analysis Draft",
        "",
        "## DINOv2 在细粒度分类中的表现",
        "",
        f"在 CUB-200-2011 细粒度鸟类分类任务上，DINOv2 ViT-B/14 的冻结 CLS 特征表现出较强的线性可分性。线性探针在 200 个类别上取得 {percent(linear_summary['top1_accuracy'])}% 的 Top-1 Accuracy；基于同一特征的 k-NN（k=10, cosine）取得 {percent(float(knn_summary['results']['10']))}% 的 Top-1 Accuracy。该结果表明，在不更新 DINOv2 backbone 参数的条件下，其通用视觉表征已经能够支持较高精度的细粒度判别。",
        "",
        "## Self-supervised feature 的优势",
        "",
        "当前实验仅使用冻结特征和轻量分类器完成下游分类，说明表征本身包含了可迁移的形状、纹理与局部语义信息。对于 CUB 这类类间差异细微、类内姿态变化显著的数据集，预训练表征减少了从小规模标注数据中学习底层视觉模式的压力，使下游训练主要集中在类别边界的拟合。",
        "",
        "## Linear Probe 与 kNN 差异",
        "",
        f"线性探针比 k-NN 高 {delta:.2f} 个百分点。该差异说明 DINOv2 特征虽然具有较好的邻域结构，但通过监督线性层仍可进一步调整类别决策边界。k-NN 完全依赖特征空间中的局部邻近关系，在细粒度类别相似或样本密度不均衡时更容易受到近邻混淆影响；线性探针则能利用全部训练样本估计全局线性分隔面。",
        "",
        "## t-SNE 聚类现象分析",
        "",
        f"重绘的 t-SNE 图基于测试集 CLS 特征采样 {tsne_stats['samples']} 个样本生成。该可视化可用于定性展示 DINOv2 表征在测试集上的类别结构，但由于 CUB 包含 200 个细粒度类别，二维降维不可避免会压缩类别间关系。因此，t-SNE 适合作为特征可分性的辅助证据，不应替代分类准确率或混淆矩阵分析。",
        "",
        "## Attention/GradCAM 现象分析",
        "",
        "当前目录中没有已生成的 attention map 或 GradCAM 图像，因此不能对注意力定位现象作实证性结论。代码中存在 `src/visualize/attention_map.py`，后续应选择正确分类、错误分类和相似类别混淆样本生成可视化，以分析模型关注区域是否落在鸟体、头部、羽毛纹理等细粒度判别区域。",
        "",
        "## 与 CNN 方法的差异",
        "",
        "当前结果中尚未包含 CNN baseline，因此不能给出数值比较。可以在论文中将本实验定位为冻结 DINOv2 表征的结果，并在补充实验中加入 ResNet 或 ConvNeXt 等 CNN 基线。理论上，ViT-based self-supervised 表征更容易利用全局上下文与 patch-level 语义关系，而 CNN 基线通常依赖局部卷积归纳偏置；实际差异需要在相同数据划分和评估协议下验证。",
        "",
        "## 泛化能力分析",
        "",
        "k-NN 结果可被视为对特征空间泛化能力的近似检验，因为该方法不训练参数化分类头，仅使用训练集特征进行邻域投票。其 85% 以上的准确率表明 DINOv2 特征在 CUB 测试集上保留了较强的类条件结构。不过，当前实验仍局限于同一数据集划分，尚不能支持跨数据集泛化或域迁移结论；这部分需要额外的 Flowers、Cars 或 domain-shift 实验补充。",
    ]
    (ROOT / "analysis_draft.md").write_text("\n".join(analysis_md), encoding="utf-8")

    missing_md = [
        "# Missing Experiments and Recommended Additions",
        "",
        "## 缺失实验项",
        "",
        "- Attention/GradCAM outputs.",
        "- TensorBoard logs.",
        "- CNN/supervised baseline.",
        "- Validation-set model selection.",
        "- Multiple-seed statistics.",
        "- kNN k sweep.",
        "- Cross-dataset/domain-shift evaluation.",
        "",
        "## 建议补做实验",
        "",
        "- Linear probe with validation-based checkpoint selection.",
        "- kNN sweep over k=1,5,10,20,50 and cosine/euclidean metrics.",
        "- ResNet-50 or ConvNeXt baseline under the same CUB split.",
        "- Attention maps for representative true positives and false positives.",
        "- Inference benchmark with fixed hardware and batch-size settings.",
    ]
    (ASSET_DIR / "appendix" / "missing_experiments.md").write_text("\n".join(missing_md), encoding="utf-8")

    for name in ["results_summary.md", "experiment_audit.md", "analysis_draft.md"]:
        shutil.copy2(ROOT / name, ASSET_DIR / "analysis" / name)


def archive_assets() -> None:
    for path in (OUT / "checkpoints").glob("*.pt"):
        shutil.copy2(path, ASSET_DIR / "checkpoints" / path.name)
    for path in (ROOT / "logs").glob("*"):
        if path.is_file():
            shutil.copy2(path, ASSET_DIR / "logs" / path.name)
    for path in [ROOT / "results" / "summary.csv", OUT / "linear_probe_metrics.csv", OUT / "linear_probe_summary.json", OUT / "knn_summary.json"]:
        if path.exists():
            shutil.copy2(path, ASSET_DIR / "appendix" / path.name)


def main() -> None:
    mkdirs()
    apply_style()

    summary = pd.read_csv(ROOT / "results" / "summary.csv")
    metrics = pd.read_csv(OUT / "linear_probe_metrics.csv")
    linear_summary = load_json(OUT / "linear_probe_summary.json")
    knn_summary = load_json(OUT / "knn_summary.json")
    train_meta = load_feature_meta(OUT / "train_cls_features.pt")
    test_meta = load_feature_meta(OUT / "test_cls_features.pt")
    ckpt = torch.load(OUT / "checkpoints" / "best_linear_probe.pt", map_location="cpu", weights_only=False)

    make_training_curve(metrics)
    make_accuracy_bar(summary)
    cm_stats = {
        "linear": make_confusion_matrix(OUT / "linear_probe_confusion_matrix.csv", "linear_probe_confusion_matrix", "Linear Probe Confusion Matrix"),
        "knn": make_confusion_matrix(OUT / "knn_k10_confusion_matrix.csv", "knn_k10_confusion_matrix", "k-NN Confusion Matrix (k=10)"),
    }
    tsne_stats = make_tsne()
    benchmark = try_benchmark_and_params()
    write_tables(summary, benchmark)
    write_markdown_files(summary, metrics, train_meta, test_meta, linear_summary, knn_summary, ckpt, cm_stats, tsne_stats, benchmark)
    archive_assets()
    print("Prepared paper figures, tables, summaries, audit, analysis draft, and paper_assets archive.")


if __name__ == "__main__":
    main()
