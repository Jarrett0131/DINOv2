#!/bin/bash

set -euo pipefail

########################################
# Init
########################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs scripts src/data
LOG_FILE="$PROJECT_ROOT/logs/download.log"

echo "Starting dataset download..." | tee -a "$LOG_FILE"

########################################
# Install dependencies
########################################

if ! command -v tree &> /dev/null
then
    echo "Installing tree..." | tee -a "$LOG_FILE"
    sudo apt update 2>&1 | tee -a "$LOG_FILE"
    sudo apt install -y tree 2>&1 | tee -a "$LOG_FILE"
fi

if ! command -v wget &> /dev/null
then
    echo "Installing wget..." | tee -a "$LOG_FILE"
    sudo apt update 2>&1 | tee -a "$LOG_FILE"
    sudo apt install -y wget 2>&1 | tee -a "$LOG_FILE"
fi

if ! command -v python3 &> /dev/null
then
    echo "Installing python3..." | tee -a "$LOG_FILE"
    sudo apt update 2>&1 | tee -a "$LOG_FILE"
    sudo apt install -y python3 python3-pip 2>&1 | tee -a "$LOG_FILE"
fi

python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import importlib.util
import subprocess
import sys

packages = {
    "datasets": "datasets>=2.18.0",
    "PIL": "Pillow>=10.0",
}

missing = [pkg for module, pkg in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

########################################
# Retry Functions
########################################

retry_wget() {
    url="$1"
    output="${2:-}"
    retries=3

    for ((i=1;i<=retries;i++)); do
        if [ -n "$output" ]; then
            wget -c -O "$output" "$url" 2>&1 | tee -a "$LOG_FILE" && return 0
        else
            wget -c "$url" 2>&1 | tee -a "$LOG_FILE" && return 0
        fi
        echo "Retry $i failed for $url" | tee -a "$LOG_FILE"
        sleep 2
    done

    echo "Download failed after $retries retries: $url" | tee -a "$LOG_FILE"
    return 1
}

extract_tgz() {
    archive="$1"
    dest="$2"
    echo "Extracting $archive -> $dest" | tee -a "$LOG_FILE"
    tar -xzf "$archive" -C "$dest" 2>&1 | tee -a "$LOG_FILE"
    rm -f "$archive"
}

########################################
# 1. CUB-200-2011
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Downloading CUB-200-2011..." | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

mkdir -p src/data/CUB_200_2011
if [ ! -d src/data/CUB_200_2011/CUB_200_2011 ]; then
    retry_wget \
        https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz \
        src/data/CUB_200_2011/CUB_200_2011.tgz
    extract_tgz src/data/CUB_200_2011/CUB_200_2011.tgz src/data/CUB_200_2011
else
    echo "CUB-200-2011 already exists. Skipping." | tee -a "$LOG_FILE"
fi

########################################
# 2. Stanford Cars
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Downloading Stanford Cars..." | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

mkdir -p src/data/StanfordCars
if [ ! -d src/data/StanfordCars/stanford-cars ]; then
    # The original Stanford server can intermittently return HTTP 500.
    # Use the public fast.ai mirror for reliable full dataset download.
    retry_wget \
        https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz \
        src/data/StanfordCars/stanford-cars.tgz
    extract_tgz src/data/StanfordCars/stanford-cars.tgz src/data/StanfordCars
else
    echo "Stanford Cars already exists. Skipping." | tee -a "$LOG_FILE"
fi

########################################
# 3. Oxford Flowers 102
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Downloading Oxford Flowers 102..." | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

mkdir -p src/data/Flowers102
if [ ! -d src/data/Flowers102/jpg ]; then
    retry_wget \
        https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz \
        src/data/Flowers102/102flowers.tgz
    retry_wget \
        https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat \
        src/data/Flowers102/imagelabels.mat
    retry_wget \
        https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat \
        src/data/Flowers102/setid.mat
    extract_tgz src/data/Flowers102/102flowers.tgz src/data/Flowers102
else
    echo "Oxford Flowers 102 already exists. Skipping image archive." | tee -a "$LOG_FILE"
    [ -f src/data/Flowers102/imagelabels.mat ] || retry_wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat src/data/Flowers102/imagelabels.mat
    [ -f src/data/Flowers102/setid.mat ] || retry_wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat src/data/Flowers102/setid.mat
fi

########################################
# 4. ImageNet-Sketch Subset
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Downloading ImageNet-Sketch subset..." | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

rm -rf src/data/ImageNetSketch
mkdir -p src/data/ImageNetSketch_subset

IMAGENET_SKETCH_NUM_CLASSES="${IMAGENET_SKETCH_NUM_CLASSES:-50}"
IMAGENET_SKETCH_IMAGES_PER_CLASS="${IMAGENET_SKETCH_IMAGES_PER_CLASS:-50}"
export IMAGENET_SKETCH_NUM_CLASSES
export IMAGENET_SKETCH_IMAGES_PER_CLASS

python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

root = Path("src/data/ImageNetSketch_subset")
root.mkdir(parents=True, exist_ok=True)

num_classes = int(os.environ.get("IMAGENET_SKETCH_NUM_CLASSES", "50"))
images_per_class = int(os.environ.get("IMAGENET_SKETCH_IMAGES_PER_CLASS", "50"))

dataset = load_dataset("imagenet_sketch", split="train", streaming=True)
label_names = None
try:
    label_names = dataset.features["label"].names
except Exception:
    label_names = None

def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("_") or "unknown"

selected_labels = []
selected_names = {}
selected_dirs = {}
counts = defaultdict(int)

for example in dataset:
    label = int(example["label"])
    if label not in selected_names:
        if len(selected_labels) >= num_classes:
            continue
        selected_labels.append(label)
        if label_names and label < len(label_names):
            class_name = label_names[label]
        else:
            class_name = f"class_{len(selected_labels)}"
        selected_names[label] = safe_name(class_name)
        selected_dirs[label] = f"class_{len(selected_labels)}"
        (root / selected_dirs[label]).mkdir(parents=True, exist_ok=True)

    if label not in selected_names:
        continue
    if counts[label] >= images_per_class:
        continue

    image = example["image"].convert("RGB")
    file_name = root / selected_dirs[label] / f"{counts[label]:05d}.jpg"
    image.save(file_name, quality=95)
    counts[label] += 1

    if len(selected_labels) >= num_classes and all(counts[l] >= images_per_class for l in selected_labels):
        break

with (root / "subset_classes.txt").open("w", encoding="utf-8") as f:
    for idx, label in enumerate(selected_labels, start=1):
        f.write(f"class_{idx}\tlabel_id={label}\tname={selected_names[label]}\timages={counts[label]}\n")

class_count = len(selected_labels)
image_count = sum(counts.values())
print(f"ImageNet-Sketch subset classes: {class_count}")
print(f"ImageNet-Sketch subset images: {image_count}")
if class_count < num_classes:
    raise RuntimeError(f"Only collected {class_count} classes; expected {num_classes}.")
PY

########################################
# Dataset Size Summary
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Dataset Size Summary" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

du -sh src/data/CUB_200_2011 src/data/StanfordCars src/data/Flowers102 src/data/ImageNetSketch_subset | tee -a "$LOG_FILE"

echo "=================================" | tee -a "$LOG_FILE"
echo "ImageNet-Sketch Subset Statistics" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

subset_classes=$(find src/data/ImageNetSketch_subset -mindepth 1 -maxdepth 1 -type d | wc -l)
subset_images=$(find src/data/ImageNetSketch_subset -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
subset_size=$(du -sh src/data/ImageNetSketch_subset | awk '{print $1}')

echo "Classes: $subset_classes" | tee -a "$LOG_FILE"
echo "Images: $subset_images" | tee -a "$LOG_FILE"
echo "Size: $subset_size" | tee -a "$LOG_FILE"

########################################
# Generate Dataset README
########################################

cat <<EOF > src/data/README_DATASETS.md
# Dataset Summary

All datasets are stored under \`src/data/\`.

## 1. CUB-200-2011
- Directory: \`src/data/CUB_200_2011/\`
- Classes: 200
- Images: 11,788
- Usage: fine-grained classification

## 2. Stanford Cars
- Directory: \`src/data/StanfordCars/\`
- Classes: 196
- Images: 16,185
- Usage: fine-grained classification validation
- Source: fast.ai public AWS mirror, used because the original Stanford server can return HTTP 500.

## 3. Oxford Flowers 102
- Directory: \`src/data/Flowers102/\`
- Classes: 102
- Images: 8,189
- Usage: k-NN zero-shot evaluation

## 4. ImageNet-Sketch Subset
- Directory: \`src/data/ImageNetSketch_subset/\`
- Classes: $subset_classes
- Images: $subset_images
- Size: $subset_size
- Usage: domain shift robustness evaluation
- Class list: \`src/data/ImageNetSketch_subset/subset_classes.txt\`

Due to course project resource limits, we use an ImageNet-Sketch subset for domain shift evaluation.

# Scenario Mapping

## Scenario 1
- CUB-200-2011
- Stanford Cars

## Scenario 2
- Oxford Flowers 102

## Scenario 3
- ImageNet-Sketch subset
EOF

########################################
# Print Directory Structure
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Dataset Directory Structure" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

tree -L 2 src/data | tee -a "$LOG_FILE"

########################################
# Finished
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "All datasets downloaded successfully!" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
