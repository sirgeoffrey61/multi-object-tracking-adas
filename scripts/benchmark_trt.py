"""Benchmark PyTorch vs TensorRT YOLOv8 inference on a sample KITTI image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def benchmark(
    model_path: Path,
    image_path: Path,
    *,
    device: str = "0",
    passes: int = 100,
    warmup: int = 10,
    label: str = "pytorch",
) -> dict:
    model = YOLO(str(model_path))

    for _ in range(warmup):
        model.predict(str(image_path), device=device, verbose=False)

    start = time.perf_counter()
    for _ in range(passes):
        model.predict(str(image_path), device=device, verbose=False)
    elapsed = time.perf_counter() - start

    fps = passes / elapsed if elapsed > 0 else 0.0
    return {
        "backend": label,
        "model": str(model_path),
        "passes": passes,
        "warmup": warmup,
        "total_sec": round(elapsed, 4),
        "fps": round(fps, 2),
        "latency_ms": round(1000.0 / fps, 3) if fps > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT inference")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/kitti_yolov8n/weights/best.pt",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="deployment/jetson_orin/best.engine",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/kitti/training/image_2/000001.png",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--passes", type=int, default=100)
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
    )
    args = parser.parse_args()

    root = project_root()
    weights = root / args.weights
    engine = root / args.engine
    image = root / args.image
    output = root / args.output

    if not image.is_file():
        raise FileNotFoundError(f"Sample image not found: {image}")

    results: dict = {
        "image": str(image),
        "device": args.device,
        "passes": args.passes,
        "pytorch": None,
        "tensorrt": None,
        "speedup": None,
    }

    if weights.is_file():
        print(f"Benchmarking PyTorch: {weights}")
        pt = benchmark(
            weights,
            image,
            device=args.device,
            passes=args.passes,
            label="pytorch",
        )
        results["pytorch"] = pt
        print(f"  PyTorch FPS: {pt['fps']:.2f}  (latency {pt['latency_ms']:.2f} ms)")
    else:
        print(f"PyTorch weights not found: {weights} — skipping")

    if engine.is_file():
        print(f"Benchmarking TensorRT: {engine}")
        trt = benchmark(
            engine,
            image,
            device=args.device,
            passes=args.passes,
            label="tensorrt",
        )
        results["tensorrt"] = trt
        print(f"  TensorRT FPS: {trt['fps']:.2f}  (latency {trt['latency_ms']:.2f} ms)")
    else:
        print(f"TensorRT engine not found: {engine} — skipping")
        print("Export on Jetson with: bash deployment/jetson_orin/export_trt.sh")

    if results["pytorch"] and results["tensorrt"]:
        speedup = results["tensorrt"]["fps"] / results["pytorch"]["fps"]
        results["speedup"] = round(speedup, 2)
        print(f"\nSpeedup (TensorRT / PyTorch): {speedup:.2f}x")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
