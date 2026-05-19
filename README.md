# DINOv2 冻结特征下游评估框架

本项目用于复现 DINOv2 的下游使用方式：不重新预训练 DINOv2，不做 full fine-tuning，只加载预训练 ViT backbone，冻结骨干网络，提取 CLS token 作为图像级特征，并在下游任务上完成评估。

项目覆盖三个实验场景：

1. 细粒度分类：CUB-200-2011 / Stanford Cars + Linear Probe。
2. k-NN 零样本/无训练迁移：Oxford Flowers 102 / CUB + cosine k-NN。
3. 域偏移鲁棒性：源域 linear probe 或 k-NN 特征库直接测试 ImageNet-Sketch subset 等目标域。

## 环境安装

```bash
pip install -r requirements.txt
```

默认 backbone 为 `dinov2_vitb14`，通过 `torch.hub` 从 `facebookresearch/dinov2` 加载。首次运行会自动下载预训练权重。

## 下载数据集

所有数据集统一放在 `src/data/` 下：

```bash
bash scripts/download_datasets.sh
```

下载完成后的目标结构：

```text
src/data/
  CUB_200_2011/
  StanfordCars/
  Flowers102/
  ImageNetSketch_subset/
    class_1/
    class_2/
    ...
    class_50/
    subset_classes.txt
  README_DATASETS.md
```

下载日志保存在：

```text
logs/download.log
```

脚本会自动创建目录、断点续传下载、失败重试、解压、删除压缩包、统计数据集大小，并打印目录结构。

由于课程项目资源限制，我们使用 ImageNet-Sketch subset 进行 domain shift evaluation。脚本只保留前 50 个类别，并生成：

```text
src/data/ImageNetSketch_subset/subset_classes.txt
```

## 数据集说明

下载脚本会保留官方原始数据结构。为了直接运行本项目的 `imagefolder` 评估入口，推荐后续整理成如下结构：

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  test/
    class_a/*.jpg
    class_b/*.jpg
```

域偏移实验推荐结构：

```text
source_root/train/class_x/*.jpg
source_root/test/class_x/*.jpg
target_root/test/class_x/*.jpg
```

ImageNet-Sketch 在本项目中只使用 subset evaluation，避免一次性下载和评估完整 ImageNet benchmark。

## 场景一：细粒度分类

提取 CUB 或 Stanford Cars 的 train/test 特征：

```bash
bash scripts/run_extract.sh imagefolder src/data/CUB_200_2011 outputs/cub
```

训练并评估 Linear Probe：

```bash
bash scripts/run_linear_probe.sh imagefolder src/data/CUB_200_2011 outputs/cub
```

直接运行 Python 命令：

```bash
python -m src.eval.linear_probe \
  --dataset imagefolder \
  --data_root src/data/CUB_200_2011 \
  --model_name dinov2_vitb14 \
  --batch_size 32 \
  --output_dir outputs/cub \
  --device cuda
```

输出包括 Top-1 Accuracy、confusion matrix 和分类结果表。

## 场景二：k-NN 零样本/无训练迁移

使用 train split 构建特征库，不训练分类头，对 test split 做 cosine k-NN：

```bash
bash scripts/run_knn.sh imagefolder src/data/Flowers102 outputs/flowers_knn
```

默认比较 `k=1,5,10,20`：

```bash
python -m src.eval.knn_eval \
  --dataset imagefolder \
  --data_root src/data/Flowers102 \
  --model_name dinov2_vitb14 \
  --batch_size 32 \
  --output_dir outputs/flowers_knn \
  --device cuda \
  --k_values 1 5 10 20
```

## 场景三：域偏移鲁棒性

在源域训练 linear probe 或构建 k-NN 特征库，然后直接测试目标域：

```bash
bash scripts/run_domain_shift.sh imagefolder /path/to/source /path/to/target outputs/domain_shift
```

直接运行：

```bash
python -m src.eval.domain_shift_eval \
  --dataset imagefolder \
  --source_root /path/to/source \
  --target_root src/data/ImageNetSketch_subset \
  --model_name dinov2_vitb14 \
  --batch_size 32 \
  --output_dir outputs/domain_shift \
  --device cuda \
  --method linear
```

脚本会报告源域 accuracy、目标域 accuracy 和 accuracy drop。

## 可视化

t-SNE：

```bash
python -m src.visualize.tsne \
  --features outputs/cub/test_cls_features.pt \
  --output outputs/cub/tsne.png
```

Attention Map：

```bash
python -m src.visualize.attention_map \
  --image /path/to/image.jpg \
  --model_name dinov2_vitb14 \
  --output outputs/attention_map.png \
  --device cuda
```

Patch Feature PCA：

```bash
python -m src.visualize.attention_map \
  --image /path/to/image.jpg \
  --output outputs/patch_pca.png \
  --mode patch_pca
```

## 输出文件

特征文件：

- `train_cls_features.pt`
- `test_cls_features.pt`
- 使用 `--save_patch_tokens` 时生成 `train_patch_tokens.pt` / `test_patch_tokens.pt`

评估文件：

- `linear_probe_metrics.csv`
- `linear_probe_confusion_matrix.csv`
- `linear_probe_report.csv`
- `knn_metrics.csv`
- `domain_shift_metrics.csv`
- `results/summary.csv`

`results/summary.csv` 字段：

```text
scenario,dataset,method,backbone,accuracy,top1,k,feature_dim
```

## 注意事项

- DINOv2 backbone 始终冻结，`requires_grad=False`。
- 图像级特征默认使用 CLS token。
- 可选保存 patch tokens，用于 attention map 或 PCA 可视化。
- 支持 GPU；如果 CUDA 不可用，会自动退到 CPU。
- 使用 `--force_extract` 可以重新抽取特征。
