#!/bin/bash

set -euo pipefail

########################################
# Init
########################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="$PROJECT_ROOT/logs/download.log"

echo "Starting dataset download..." | tee -a "$LOG_FILE"

mkdir -p src/data
mkdir -p scripts

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

if ! command -v git &> /dev/null
then
    echo "Installing git..." | tee -a "$LOG_FILE"
    sudo apt update 2>&1 | tee -a "$LOG_FILE"
    sudo apt install -y git 2>&1 | tee -a "$LOG_FILE"
fi

########################################
# Retry Functions
########################################

retry_wget() {
    url="$1"
    retries=3

    for ((i=1;i<=retries;i++)); do
        if wget -c "$url" 2>&1 | tee -a "$LOG_FILE"; then
            return 0
        fi
        echo "Retry $i failed for $url" | tee -a "$LOG_FILE"
        sleep 2
    done

    echo "Download failed after $retries retries: $url" | tee -a "$LOG_FILE"
    return 1
}

retry_git_clone() {
    url="$1"
    target="$2"
    retries=3

    if [ -d "$target/.git" ]; then
        echo "$target already exists, pulling latest changes..." | tee -a "$LOG_FILE"
        git -C "$target" pull 2>&1 | tee -a "$LOG_FILE"
        return 0
    fi

    for ((i=1;i<=retries;i++)); do
        if git clone "$url" "$target" 2>&1 | tee -a "$LOG_FILE"; then
            return 0
        fi
        echo "Retry $i failed for git clone $url" | tee -a "$LOG_FILE"
        rm -rf "$target"
        sleep 2
    done

    echo "Git clone failed after $retries retries: $url" | tee -a "$LOG_FILE"
    return 1
}

########################################
# 1. CUB-200-2011
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Downloading CUB-200-2011..." | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

mkdir -p src/data/CUB_200_2011
cd src/data/CUB_200_2011

retry_wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

tar -xvf CUB_200_2011.tgz 2>&1 | tee -a "$LOG_FILE"

rm CUB_200_2011.tgz

cd "$PROJECT_ROOT"

########################################
# 2. Stanford Cars
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Downloading Stanford Cars..." | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

mkdir -p src/data/StanfordCars
cd src/data/StanfordCars

retry_wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
retry_wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
retry_wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
retry_wget https://ai.stanford.edu/~jkrause/cars/cars_test_annos_withlabels.mat

tar -xvf cars_train.tgz 2>&1 | tee -a "$LOG_FILE"
tar -xvf cars_test.tgz 2>&1 | tee -a "$LOG_FILE"
tar -xvf car_devkit.tgz 2>&1 | tee -a "$LOG_FILE"

rm *.tgz

cd "$PROJECT_ROOT"

########################################
# 3. Oxford Flowers 102
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Downloading Oxford Flowers 102..." | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

mkdir -p src/data/Flowers102
cd src/data/Flowers102

retry_wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
retry_wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
retry_wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

tar -xvf 102flowers.tgz 2>&1 | tee -a "$LOG_FILE"

rm 102flowers.tgz

cd "$PROJECT_ROOT"

########################################
# 4. ImageNet-Sketch
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Downloading ImageNet-Sketch..." | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

retry_git_clone https://github.com/HaohanWang/ImageNet-Sketch.git src/data/ImageNetSketch

########################################
# Dataset Size Summary
########################################

echo "=================================" | tee -a "$LOG_FILE"
echo "Dataset Size Summary" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

du -sh src/data/* | tee -a "$LOG_FILE"

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
- Usage:
  - Fine-grained classification

## 2. Stanford Cars
- Directory: \`src/data/StanfordCars/\`
- Classes: 196
- Images: 16,185
- Usage:
  - Fine-grained classification validation

## 3. Oxford Flowers 102
- Directory: \`src/data/Flowers102/\`
- Classes: 102
- Images: 8,189
- Usage:
  - k-NN zero-shot evaluation

## 4. ImageNet-Sketch
- Directory: \`src/data/ImageNetSketch/\`
- Classes: 1000
- Usage:
  - Domain shift robustness evaluation
  - Subset evaluation is recommended for faster experiments

# Scenario Mapping

## Scenario 1
- CUB-200-2011
- Stanford Cars

## Scenario 2
- Oxford Flowers 102

## Scenario 3
- ImageNet-Sketch
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
