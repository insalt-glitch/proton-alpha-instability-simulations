#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target-directory>"
    exit 1
fi

ROOT_DIR="$1"

pixi run python tools/sdf2h5.py --no-archive -r $ROOT_DIR
# pixi run python tools/h5vstack.py -p 1 -r "$ROOT_DIR"
pixi run python u_alpha_dispersion.py
