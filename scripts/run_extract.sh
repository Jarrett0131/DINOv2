#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-imagefolder}"
DATA_ROOT="${2:-data}"
OUTPUT_DIR="${3:-outputs/extract}"
MODEL_NAME="${MODEL_NAME:-dinov2_vitb14}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"

python -m src.features.extract_features \
  --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --save_patch_tokens
