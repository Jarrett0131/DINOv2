# DINOv2 Frozen Feature Evaluation

本项目复现 DINOv2 的下游使用方式：不重新预训练，不 full fine-tuning，只加载预训练
DINOv2 ViT backbone，冻结参数，提取 CLS token 特征，并在下游任务上做评估。

覆盖三个实验场景：

1. 细粒度分类：CUB-200-2011 / Stanford Cars + Linear Probe。
2. k-NN 零样本/无训练迁移：Oxford Flowers 102 / CUB + cosine k-NN。
3. 域偏移鲁棒性：源域训练好的 linear probe 或 k-NN 特征库直接测试 Sketch/Cartoon 目标域。

## Install

```bash
pip install -r requirements.txt
```

默认模型是 `dinov2_vitb14`，通过 `torch.hub` 从 `facebookresearch/dinov2` 加载。首次运行会下载权重。

## Dataset Layout

推荐把 CUB、Cars、ImageNet-Sketch 子集或自定义数据整理成 `ImageFolder` 结构：

```text
data_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  test/
    class_a/*.jpg
    class_b/*.jpg
```

域偏移实验需要源域和目标域两个根目录：

```text
source_root/train/class_x/*.jpg
source_root/test/class_x/*.jpg
target_root/test/class_x/*.jpg
```

## Scenario 1: Fine-Grained Classification

提取 CUB 或 Stanford Cars 的 train/test 特征：

```bash
bash scripts/run_extract.sh imagefolder /path/to/CUB outputs/cub
```

训练 Linear Probe 并报告 Top-1 Accuracy、confusion matrix 和分类结果表：

```bash
bash scripts/run_linear_probe.sh imagefolder /path/to/CUB outputs/cub
```

直接运行：

```bash
python -m src.eval.linear_probe \
  --dataset imagefolder \
  --data_root /path/to/CUB \
  --model_name dinov2_vitb14 \
  --batch_size 32 \
  --output_dir outputs/cub \
  --device cuda
```

## Scenario 2: k-NN Zero-Shot / No-Training Transfer

不训练分类头，只用 train split 构建特征库，对 test split 做 cosine k-NN：

```bash
bash scripts/run_knn.sh imagefolder /path/to/flowers outputs/flowers_knn
```

默认比较 `k=1,5,10,20`：

```bash
python -m src.eval.knn_eval \
  --dataset imagefolder \
  --data_root /path/to/flowers \
  --model_name dinov2_vitb14 \
  --batch_size 32 \
  --output_dir outputs/flowers_knn \
  --device cuda \
  --k_values 1 5 10 20
```

## Scenario 3: Domain Shift Robustness

源域训练 linear probe，直接测试目标域：

```bash
bash scripts/run_domain_shift.sh imagefolder /path/to/source /path/to/target outputs/domain_shift
```

直接运行：

```bash
python -m src.eval.domain_shift_eval \
  --dataset imagefolder \
  --source_root /path/to/source \
  --target_root /path/to/target \
  --model_name dinov2_vitb14 \
  --batch_size 32 \
  --output_dir outputs/domain_shift \
  --device cuda \
  --method linear
```

使用 k-NN 特征库做域偏移：

```bash
python -m src.eval.domain_shift_eval \
  --dataset imagefolder \
  --source_root /path/to/source \
  --target_root /path/to/target \
  --output_dir outputs/domain_shift_knn \
  --method knn \
  --k 10
```

输出包括 `source_accuracy`、`target_accuracy` 和 `accuracy_drop`。

## Visualization

t-SNE：

```bash
python -m src.visualize.tsne \
  --features outputs/cub/test_cls_features.pt \
  --output outputs/cub/tsne.png
```

Attention map：

```bash
python -m src.visualize.attention_map \
  --image /path/to/image.jpg \
  --model_name dinov2_vitb14 \
  --output outputs/attention_map.png \
  --device cuda
```

Patch feature PCA：

```bash
python -m src.visualize.attention_map \
  --image /path/to/image.jpg \
  --output outputs/patch_pca.png \
  --mode patch_pca
```

## Outputs

特征文件：

- `train_cls_features.pt`
- `test_cls_features.pt`
- `train_patch_tokens.pt` / `test_patch_tokens.pt` when `--save_patch_tokens` is set

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

## Notes

- Backbone 始终冻结，`requires_grad=False`。
- 图像级特征使用 CLS token。
- 支持 GPU；当 `--device cuda` 但 CUDA 不可用时会自动退到 CPU。
- 可通过 `--force_extract` 重新抽取特征。
- 可视化命令需要已有特征文件或单张输入图像。仓库提供代码入口；实际图片由你的数据集运行后生成。
