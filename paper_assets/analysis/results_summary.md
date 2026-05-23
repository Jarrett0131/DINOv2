# Results Summary

## 核心指标

- Dataset: CUB-200-2011; train=5994, test=5794, classes=200.
- Backbone: dinov2_vitb14; image size=224; feature dimension=768.
- Linear Probe Top-1 Accuracy: 88.45%; best epoch=14.
- k-NN Top-1 Accuracy: 85.57%; k=10, metric=cosine.
- Train/test curve: epochs 6-20; best test accuracy at epoch 14.
- Inference speed: 32.56 images/s (Measured on 20 test batches after 2 warmup batches; includes image transfer and backbone forward only.).

## 文件清单与论文可用性

| 项目 | 文件路径 | 文件作用 | 是否可直接用于论文 | 是否需要重跑 |
|---|---|---|---|---|
| Linear Probe Accuracy | `outputs/cub_dinov2_vitb14/linear_probe_summary.json` | linear probe top-1, best checkpoint epoch and feature dimension | 是 | 否 |
| k-NN Accuracy | `outputs/cub_dinov2_vitb14/knn_summary.json` | k=10 cosine kNN top-1 | 是 | 否 |
| train/test loss 曲线 | `outputs/cub_dinov2_vitb14/linear_probe_metrics.csv` | epoch-level train/test loss and accuracy | 是，已重绘 | 否 |
| confusion matrix | `outputs/cub_dinov2_vitb14/linear_probe_confusion_matrix.csv` | linear probe 200x200 confusion matrix | 是，已重绘 | 否 |
| confusion matrix | `outputs/cub_dinov2_vitb14/knn_k10_confusion_matrix.csv` | kNN 200x200 confusion matrix | 是，已重绘 | 否 |
| t-SNE 可视化 | `outputs/cub_dinov2_vitb14/tsne/test_tsne.png` | test-set DINOv2 CLS t-SNE legacy image | 原图不建议；已重绘 | 否 |
| attention map / gradcam | `src/visualize/attention_map.py` | has visualization script, no generated output found | 否 | 建议补做 |
| 推理速度 | `paper_tables/csv/inference_speed.csv` | lightweight benchmark result or explicit missing note | 若有数值则可用；否则需补测 | 视 benchmark 结果 |
| 参数量 | `paper_tables/csv/parameter_comparison.csv` | backbone/head parameter count | 是（若 backbone 成功测量） | 否 |
| 特征维度 | `outputs/cub_dinov2_vitb14/train_cls_features.pt` | feature tensor metadata, dim=768 | 是 | 否 |
| checkpoint | `outputs/cub_dinov2_vitb14/checkpoints/best_linear_probe.pt` | best linear probe classifier checkpoint | 可复现实验；论文不直接展示 | 否 |

## 图表筛查结果

- Legacy t-SNE PNG is 1600x1400 and was saved at 200 dpi; it uses a repeated `tab20` palette for 200 classes, lacks explicit x/y axis labels, and has no PDF companion. It is not recommended as the submitted figure.
- Confusion matrices existed only as CSV files, so they were not directly usable as figures before this pass.
- Training curves existed only as CSV/log text and were not directly usable before this pass.
- No attention map or GradCAM output image was found under the scanned result directories.

## 已重绘论文图表

- `paper_figures/tsne_test_features.png` and `.pdf`
- `paper_figures/linear_probe_confusion_matrix.png` and `.pdf`
- `paper_figures/knn_k10_confusion_matrix.png` and `.pdf`
- `paper_figures/train_test_curve.png` and `.pdf`
- `paper_figures/accuracy_comparison.png` and `.pdf`

## 缺失实验项列表

- Attention/GradCAM figure outputs are missing.
- TensorBoard event logs are missing.
- Independent CNN baseline results are missing.
- Cross-dataset/domain-shift results are not present, although a domain shift evaluation module exists.
- kNN hyperparameter sweep beyond k=10 is missing.
- Multiple seeds or confidence intervals are missing.
- Paper-grade inference benchmark is missing if the lightweight CUDA benchmark did not produce a value.