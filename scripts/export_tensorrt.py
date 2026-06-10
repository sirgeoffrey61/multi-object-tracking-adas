"""Export trained YOLOv8 weights to TensorRT engine (FP16) for Jetson Orin deployment."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from ultralytics import YOLO


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


def print_tensorrt_fallback(weights: Path) -> None:
    print("\nTensorRT is not available on this machine.")
    print("This script targets NVIDIA Jetson Orin / DRIVE Orin with JetPack and TensorRT.")
    print("On device, run: bash deployment/jetson_orin/export_trt.sh")
    print(f"Or: yolo export model={weights} format=engine device=0 half=True")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to TensorRT FP16 engine")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/kitti_yolov8n/weights/best.pt",
        help="Path to trained PyTorch weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deployment/jetson_orin/best.engine",
        help="Destination path for the TensorRT engine",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    args = parser.parse_args()

    root = project_root()
    weights = root / args.weights
    output = root / args.output

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}. Train with scripts/train.py first.")

    output.parent.mkdir(parents=True, exist_ok=True)

    if not tensorrt_available():
        print_tensorrt_fallback(weights)
        sys.exit(0)

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    try:
        start = time.perf_counter()
        exported = model.export(format="engine", device=args.device, half=True)
        elapsed = time.perf_counter() - start

        exported_path = Path(exported)
        if exported_path.resolve() != output.resolve():
            shutil.copy2(exported_path, output)

        size_bytes = output.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        print("\n=== TensorRT export complete ===")
        print(f"Export time:  {elapsed:.2f}s")
        print(f"Engine size:  {size_mb:.2f} MB ({size_bytes:,} bytes)")
        print(f"Saved to:     {output}")
    except Exception as exc:
        msg = str(exc).lower()
        if "tensorrt" in msg or "nvinfer" in msg:
            print_tensorrt_fallback(weights)
            sys.exit(0)
        raise


if __name__ == "__main__":
    main()
