#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-64}"
DATASET="${1:-auto}"
DATA_ROOT="${2:-src/data/CUB_200_2011}"
OUTPUT_DIR="${3:-outputs/linear_probe}"
MODEL_NAME="${MODEL_NAME:-dinov2_vitb14}"
DEVICE="${DEVICE:-cuda}"

python -m src.eval.linear_probe \
  --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-0.001}" \
  --image_size "${IMAGE_SIZE:-224}"
