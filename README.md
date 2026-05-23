# DINOv2 Frozen Feature Evaluation

本项目用于复现“DINOv2 在细粒度分类与零样本迁移场景下的特征评估”。DINOv2 始终作为 frozen backbone，只训练线性分类头，并评估 k-NN 与 t-SNE 可视化。

## 目录结构

```text
.
├── datasets/          # 数据集构建入口，复用 src/data
├── models/            # DINOv2 backbone 入口
├── train/             # Linear Probe 训练入口
├── eval/              # k-NN 等评估入口
├── visualize/         # t-SNE 可视化入口
├── outputs/           # 实验输出
├── configs/           # 默认实验配置
├── src/               # 主要实现代码
└── main.py            # 统一实验入口
```

## 安装

```bash
pip install -r requirements.txt
```

首次运行会通过 `torch.hub` 从 `facebookresearch/dinov2` 加载 `dinov2_vitb14` 权重。

## 数据集

默认数据根目录是 `src/data`。当前代码支持：

- CUB-200-2011：官方 `images.txt`、`image_class_labels.txt`、`train_test_split.txt`
- Oxford Flowers 102：`jpg/`、`imagelabels.mat`、`setid.mat`
- Stanford Cars：需要 `cars_train`、`cars_test` 以及官方 `.mat` 标注文件
- 普通 ImageFolder：`root/train/class/*.jpg` 与 `root/test/class/*.jpg`

所有图像默认 resize 到 `224x224`，并使用 ImageNet mean/std 归一化。

## 一键运行 baseline

CUB:

```bash
python main.py ^
  --mode all ^
  --dataset cub ^
  --data_root src/data/CUB_200_2011 ^
  --output_dir outputs/cub_dinov2_vitb14 ^
  --batch_size 64 ^
  --epochs 20 ^
  --device cuda
```

Flowers102:

```bash
python main.py ^
  --mode all ^
  --dataset flowers ^
  --data_root src/data/Flowers102 ^
  --output_dir outputs/flowers_dinov2_vitb14 ^
  --batch_size 64 ^
  --epochs 20 ^
  --device cuda
```

只跑某个阶段：

```bash
python main.py --mode linear --dataset cub --data_root src/data/CUB_200_2011 --output_dir outputs/cub
python main.py --mode knn --dataset flowers --data_root src/data/Flowers102 --output_dir outputs/flowers
python main.py --mode tsne --dataset cub --data_root src/data/CUB_200_2011 --output_dir outputs/cub
```

## 实验细节

Linear Probe:

- backbone: `dinov2_vitb14`
- feature: CLS token, shape `[batch, feature_dim]`
- backbone: `requires_grad=False`
- classifier: `nn.Linear(feature_dim, num_classes)`
- optimizer: AdamW
- lr: `1e-3`
- epochs: `20`
- batch size: `64`
- 自动保存 best checkpoint

k-NN:

- `sklearn.neighbors.KNeighborsClassifier`
- default `k=10`
- default metric: cosine
- 使用 train CLS feature fit，test CLS feature predict

t-SNE:

- `sklearn.manifold.TSNE`
- 默认随机采样最多 2000 个 test features
- 输出 PNG 到 `outputs/.../tsne/`

## 输出说明

典型输出目录：

```text
outputs/cub_dinov2_vitb14/
├── train_cls_features.pt
├── test_cls_features.pt
├── linear_probe_metrics.csv
├── linear_probe_confusion_matrix.csv
├── linear_probe_report.csv
├── linear_probe_summary.json
├── knn_metrics.csv
├── knn_k10_confusion_matrix.csv
├── knn_k10_report.csv
├── knn_summary.json
├── checkpoints/
│   └── best_linear_probe.pt
└── tsne/
    └── test_tsne.png
```

全局汇总会追加到：

```text
results/summary.csv
```

## 快速 smoke test

为了只检查代码链路，可以限制 batch 数：

```bash
python main.py --mode linear --dataset cub --data_root src/data/CUB_200_2011 --output_dir outputs/smoke --epochs 1 --max_train_batches 1 --max_eval_batches 1
```

正式结果请去掉 `--max_train_batches` 和 `--max_eval_batches`。
