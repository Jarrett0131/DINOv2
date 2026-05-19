#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-imagefolder}"
SOURCE_ROOT="${2:-data/source}"
TARGET_ROOT="${3:-data/target}"
OUTPUT_DIR="${4:-outputs/domain_shift}"
MODEL_NAME="${MODEL_NAME:-dinov2_vitb14}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
METHOD="${METHOD:-linear}"

python -m src.eval.domain_shift_eval \
  --dataset "$DATASET" \
  --source_root "$SOURCE_ROOT" \
  --target_root "$TARGET_ROOT" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --method "$METHOD"
