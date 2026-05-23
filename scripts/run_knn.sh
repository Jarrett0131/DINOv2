#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-auto}"
DATA_ROOT="${2:-src/data/CUB_200_2011}"
OUTPUT_DIR="${3:-outputs/knn}"
MODEL_NAME="${MODEL_NAME:-dinov2_vitb14}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

python -m src.eval.knn_eval \
  --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --image_size "${IMAGE_SIZE:-224}" \
  --k "${K:-10}"
