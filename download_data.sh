#!/bin/bash
set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

# URLs for MNIST dataset (using a reliable mirror)
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"
FILES=("train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz")

echo "Downloading MNIST dataset..."

for file in "${FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "Downloading $file..."
        curl -L "$BASE_URL/$file" -o "$DATA_DIR/$file"
    else
        echo "$file already exists."
    fi
done

echo "Download complete."
