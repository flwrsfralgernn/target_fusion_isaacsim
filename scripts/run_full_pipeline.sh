#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-/home/rog/Downloads/isaacsim/python.sh}"
HOST_PYTHON="${HOST_PYTHON:-python3}"

if [[ ! -x "$ISAAC_SIM_PYTHON" ]]; then
    echo "Isaac Sim Python launcher is not executable: $ISAAC_SIM_PYTHON" >&2
    echo "Set ISAAC_SIM_PYTHON to the correct python.sh path and try again." >&2
    exit 1
fi

if ! command -v "$HOST_PYTHON" >/dev/null 2>&1; then
    echo "Host Python executable was not found: $HOST_PYTHON" >&2
    exit 1
fi

cd "$PROJECT_DIR"

echo "[1/5] Capturing 2,000 episodes (approximately 8,000 camera images)..."
"$ISAAC_SIM_PYTHON" scripts/cycle_ground_backgrounds.py \
    --headless \
    --frames 2000 \
    --sensor-noise \
    --seed 42

echo "[2/5] Exporting the YOLO dataset..."
"$HOST_PYTHON" scripts/export_yolo_dataset.py \
    --schema-v2-output outputs/target_fusion_bbox_v2.jsonl \
    --output-dir outputs/yolo_mannequin \
    --overwrite

echo "[3/5] Validating the YOLO dataset..."
"$HOST_PYTHON" scripts/validate_yolo_dataset.py \
    --dataset-dir outputs/yolo_mannequin

echo "[4/5] Rendering a 16-image training preview..."
"$HOST_PYTHON" scripts/visualize_yolo_dataset.py \
    --dataset-dir outputs/yolo_mannequin \
    --split train \
    --limit 16 \
    --output-dir outputs/yolo_mannequin/previews

echo "[5/5] Training YOLO..."
"$HOST_PYTHON" scripts/train_yolo_local.py \
    --data outputs/yolo_mannequin/data.yaml

echo "Pipeline complete."
