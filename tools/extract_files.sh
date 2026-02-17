#!/usr/bin/env bash

set -euo pipefail

# ---- Check argument ----
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <download-directory> <target-directory>"
    exit 1
fi

SRC_DIR="$1"
ROOT_DIR="$2"

mkdir -p $ROOT_DIR/epoch_1D
tar -xf "$SRC_DIR/particle_number_variation.tar" -C $ROOT_DIR/epoch_1D/

# ---- Find and process files ----
find "$ROOT_DIR" -type f \( -name "0.tar.xz" -o -name "1.tar.xz" -o -name "2.tar.xz" -o -name "3.tar.xz" \) | while read -r file; do
    dir=$(dirname "$file")
    base=$(basename "$file")

    num="${base%%.tar.xz}"
    target_dir="$dir/rep_$num"
    
    # ---- Skip if directory already exists ----
    if [ -d "$target_dir" ]; then
        echo "Skipping $file (target exists: $target_dir)"
        continue
    fi

    mkdir -p "$target_dir"

    echo "Extracting $file -> $target_dir"
    tar -xJf "$file" -C "$target_dir"
    rm "$file"
done

mkdir -p $ROOT_DIR/epoch_2D/v_alpha_bulk_variation
# ---- Find matching archives ----
for file in $SRC_DIR/u_alpha_*kms.tar.xz; do
    # If no files match, the glob stays literal — skip safely
    [ -e "$file" ] || continue

    base=$(basename "$file")
    value=$(echo "$base" | sed -E 's/^u_alpha_([0-9]+)kms\.tar\.xz$/\1/')

    # Build destination directory
    target_dir="$ROOT_DIR/epoch_2D/v_alpha_bulk_variation/u_alpha_$value"
    
    # ---- Skip if directory already exists ----
    if [ -d "$target_dir" ]; then
        echo "Skipping $file (target exists: $target_dir)"
        continue
    fi

    mkdir -p "$target_dir"

    echo "Extracting $file -> $target_dir"

    tar -xJf "$file" -C "$target_dir"
done

file="$SRC_DIR/u_alpha_150_grid_variation.tar.xz"
target_dir="$ROOT_DIR/epoch_2D/special_grid_v_alpha_bulk_150"
if [ ! -d "$target_dir" ]; then
    mkdir -p "$target_dir"
    echo "Extracting $file -> $target_dir"
    tar -xJf "$SRC_DIR/u_alpha_150_grid_variation.tar.xz" -C "$target_dir"
fi

