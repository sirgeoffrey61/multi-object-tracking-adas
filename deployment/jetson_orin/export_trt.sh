#!/usr/bin/env bash
# Export KITTI-trained YOLOv8 weights to TensorRT FP16 on Jetson Orin / DRIVE Orin.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 scripts/export_tensorrt.py \
  --weights runs/train/kitti_yolov8n/weights/best.pt \
  --output deployment/jetson_orin/best.engine \
  --device 0
