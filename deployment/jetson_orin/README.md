# Jetson Orin / DRIVE Orin Deployment

## Target Hardware
- NVIDIA Jetson AGX Orin (275 TOPS)
- NVIDIA DRIVE Orin SoC (254 TOPS)
- JetPack 6.x / CUDA 12.x

## Pipeline Mapping
| This Project | DRIVE Hyperion Equivalent |
|---|---|
| YOLOv8 detection | Perception module |
| DeepSORT tracking | State estimation / object continuity |
| Lane detection | Scene understanding / road topology |
| FastAPI /inference | Edge inference endpoint (Orin-ready) |

## Export Steps
1. Train model (done): runs/train/kitti_yolov8n/weights/best.pt
2. Export to TensorRT: bash export_trt.sh
3. Expected speedup: 3-5x over PyTorch on Orin
4. Deploy via Docker container on Jetson

## Export Command
yolo export model=best.pt format=engine device=0 half=True

## Performance Targets
| Metric | PyTorch | TensorRT FP16 |
|--------|---------|---------------|
| FPS (Orin) | ~45 | ~150+ |
| Latency | ~22ms | ~6ms |
| Model size | 24.5 MB | ~12 MB |
