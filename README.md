# DINOv2 Frozen Feature Evaluation

This project reproduces a DINOv2 downstream evaluation workflow without pretraining:
load a pretrained DINOv2 ViT backbone, freeze it, extract CLS-token image features,
and evaluate linear probing, cosine k-NN, and domain-shift robustness.

## Install

```bash
pip install -r requirements.txt
```

The default backbone is `dinov2_vitb14`, loaded through `torch.hub` from
`facebookresearch/dinov2`. The first run may download pretrained weights.

## Download Datasets

All required datasets are downloaded and organized under `src/data/`:

```bash
bash scripts/download_datasets.sh
```

Expected dataset layout:

```text
src/data/
  CUB_200_2011/
  StanfordCars/
  Flowers102/
  ImageNetSketch/
  README_DATASETS.md
```

Download logs are written to:

```text
logs/download.log
```

The script uses `wget -c`, retries failed downloads, extracts archives, removes
downloaded archives, prints dataset sizes, and shows the final directory tree.

## Dataset Notes

The evaluation code is easiest to use with `ImageFolder`-style splits:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  test/
    class_a/*.jpg
    class_b/*.jpg
```

Some official archives, such as CUB and Stanford Cars, may need a split-conversion
step if you want direct `ImageFolder` training and testing. The download script
keeps the official raw dataset structure under `src/data/`.

For domain shift evaluation:

```text
source_root/train/class_x/*.jpg
source_root/test/class_x/*.jpg
target_root/test/class_x/*.jpg
```

ImageNet-Sketch is intended for subset evaluation in this project.

## Scenario 1: Fine-Grained Classification

Extract train/test features:

```bash
bash scripts/run_extract.sh imagefolder src/data/CUB_200_2011 outputs/cub
```

Train and evaluate a linear probe:

```bash
bash scripts/run_linear_probe.sh imagefolder src/data/CUB_200_2011 outputs/cub
```

Direct command:

```bash
python -m src.eval.linear_probe \
  --dataset imagefolder \
  --data_root src/data/CUB_200_2011 \
  --model_name dinov2_vitb14 \
  --batch_size 32 \
  --output_dir outputs/cub \
  --device cuda
```

Outputs include Top-1 accuracy, `linear_probe_confusion_matrix.csv`, and
`linear_probe_report.csv`.

## Scenario 2: k-NN Zero-Shot / No-Training Transfer

Run cosine k-NN with `k=1,5,10,20`:

```bash
bash scripts/run_knn.sh imagefolder src/data/Flowers102 outputs/flowers_knn
```

Direct command:

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

## Scenario 3: Domain Shift Robustness

Train a source-domain linear probe or build a source-domain k-NN feature bank,
then evaluate directly on the target domain:

```bash
bash scripts/run_domain_shift.sh imagefolder /path/to/source /path/to/target outputs/domain_shift
```

Direct command:

```bash
python -m src.eval.domain_shift_eval \
  --dataset imagefolder \
  --source_root /path/to/source \
  --target_root src/data/ImageNetSketch \
  --model_name dinov2_vitb14 \
  --batch_size 32 \
  --output_dir outputs/domain_shift \
  --device cuda \
  --method linear
```

The script reports source accuracy, target accuracy, and accuracy drop.

## Visualization

t-SNE:

```bash
python -m src.visualize.tsne \
  --features outputs/cub/test_cls_features.pt \
  --output outputs/cub/tsne.png
```

Attention map:

```bash
python -m src.visualize.attention_map \
  --image /path/to/image.jpg \
  --model_name dinov2_vitb14 \
  --output outputs/attention_map.png \
  --device cuda
```

Patch feature PCA:

```bash
python -m src.visualize.attention_map \
  --image /path/to/image.jpg \
  --output outputs/patch_pca.png \
  --mode patch_pca
```

## Outputs

Feature files:

- `train_cls_features.pt`
- `test_cls_features.pt`
- `train_patch_tokens.pt` / `test_patch_tokens.pt` when `--save_patch_tokens` is set

Evaluation files:

- `linear_probe_metrics.csv`
- `linear_probe_confusion_matrix.csv`
- `linear_probe_report.csv`
- `knn_metrics.csv`
- `domain_shift_metrics.csv`
- `results/summary.csv`

`results/summary.csv` columns:

```text
scenario,dataset,method,backbone,accuracy,top1,k,feature_dim
```

## Notes

- The DINOv2 backbone is always frozen with `requires_grad=False`.
- CLS token features are used as image-level representations.
- Patch tokens can optionally be saved for attention or PCA visualization.
- CUDA is supported; CPU is used automatically when CUDA is unavailable.
- Use `--force_extract` to regenerate cached features.
