# NAVI — CV / ADAS Learning Roadmap

---

## SESSION LOG

| Date | Milestone |
|------|-----------|
| June 10, 2026 | Phase 1 COMPLETE — multi-object-tracking-adas |
| | YOLOv8n trained on KITTI mAP50 0.8521, DeepSORT tracking, |
| | lane detection, FastAPI, MLflow registry, GitHub Actions CI, |
| | TensorRT export, Jetson Orin deployment folder |
| | Repo: github.com/sirgeoffrey61/multi-object-tracking-adas |
| June 11, 2026 | Phase 2 COMPLETE — depth-estimation-adas |
| | MiDaS_small + YOLOv8n, distance per object, |
| | BEV visualization, FastAPI /health /metrics, |
| | 4/4 tests passing, distance formula fixed |
| | Repo: github.com/sirgeoffrey61/depth-estimation-adas |

---

## 6-MONTH CV/ADAS ROADMAP

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Multi-object detection & tracking (KITTI) | ✅ COMPLETE |
| **Phase 2** | Edge deployment & TensorRT on Jetson Orin | ✅ COMPLETE |
| **Phase 3** | | 🔴 IN PROGRESS |

---

## COMPLETED PROJECTS

### 6. Depth Estimation for ADAS
- **Repo:** github.com/sirgeoffrey61/depth-estimation-adas
- **What it does:** Monocular depth estimation fused with 
  YOLOv8 detections — estimates distance to each object 
  from a single camera
- **Stack:** PyTorch, MiDaS, YOLOv8, OpenCV, Open3D, FastAPI
- **Status:** Complete, on GitHub ✅
