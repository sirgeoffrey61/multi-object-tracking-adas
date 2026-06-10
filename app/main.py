"""FastAPI service and real-time video inference pipeline."""

from __future__ import annotations

import base64
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.detector import ObjectDetector
from app.lane_detector import LaneDetector
from app.tracker import ObjectTracker

# ---------------------------------------------------------------------------
# Shared pipeline state
# ---------------------------------------------------------------------------

_detector: ObjectDetector | None = None
_tracker: ObjectTracker | None = None
_lane_detector: LaneDetector | None = None

# In-memory request / latency metrics (no external collector required).
_app_start_time: float = time.time()
_total_requests: int = 0
_total_inference_ms: float = 0.0

DEFAULT_MODEL_NAME = "kitti-yolov8n"
DEFAULT_MAP50 = 0.8521


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_detector() -> ObjectDetector:
    global _detector
    if _detector is None:
        model_path = os.environ.get("YOLO_MODEL_PATH")
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(_project_root(), model_path)
        _detector = ObjectDetector(model_path=model_path)
    return _detector


def _get_tracker() -> ObjectTracker:
    global _tracker
    if _tracker is None:
        embedder_gpu = os.environ.get("EMBEDDER_GPU", "0") == "1"
        _tracker = ObjectTracker(embedder_gpu=embedder_gpu)
    return _tracker


def _get_lane_detector() -> LaneDetector:
    global _lane_detector
    if _lane_detector is None:
        _lane_detector = LaneDetector()
    return _lane_detector


def _device_label() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _model_name() -> str:
    path = os.environ.get("YOLO_MODEL_PATH", "")
    if path:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem and stem != "yolov8n":
            return stem
    return os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)


def _model_map50() -> float:
    raw = os.environ.get("MODEL_MAP50")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_MAP50


def _record_inference(inference_ms: float) -> None:
    global _total_requests, _total_inference_ms
    _total_requests += 1
    _total_inference_ms += inference_ms


# ---------------------------------------------------------------------------
# Drawing & serialization
# ---------------------------------------------------------------------------

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (255, 128, 0),
    "pedestrian": (0, 0, 255),
    "cyclist": (0, 255, 128),
}


def _draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        l, t, r, b = (int(v) for v in track.bbox)
        color = CLASS_COLORS.get(track.class_name, (255, 255, 255))
        cv2.rectangle(out, (l, t), (r, b), color, 2)
        label = f"ID {track.track_id} {track.class_name}"
        cv2.putText(
            out,
            label,
            (l, max(t - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _encode_image_b64(image: np.ndarray, fmt: str = ".jpg") -> str:
    ok, buf = cv2.imencode(fmt, image)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_upload_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image bytes")
    return frame


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------


class VideoPipeline:
    """Frame-by-frame detect → track → lane overlay for files or cameras."""

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        tracker: ObjectTracker | None = None,
        lane_detector: LaneDetector | None = None,
        enable_lanes: bool = True,
    ) -> None:
        self.detector = detector or _get_detector()
        self.tracker = tracker or _get_tracker()
        self.lane_detector = lane_detector or _get_lane_detector()
        self.enable_lanes = enable_lanes

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(frame, detections)
        annotated = _draw_tracks(frame, tracks)

        lane_info: dict[str, Any] = {"left": None, "right": None}
        if self.enable_lanes:
            lanes = self.lane_detector.detect(frame)
            annotated = self.lane_detector.draw_lanes(annotated, lanes)
            lane_info = {
                "left": list(lanes.left) if lanes.left else None,
                "right": list(lanes.right) if lanes.right else None,
            }

        meta = {
            "num_detections": len(detections),
            "num_tracks": len(tracks),
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "confidence": round(t.confidence, 4),
                    "bbox": [round(v, 2) for v in t.bbox],
                }
                for t in tracks
            ],
            "lanes": lane_info,
        }
        return annotated, meta

    def run_on_capture(
        self,
        source: int | str,
        max_frames: int | None = None,
    ):
        """Generator yielding JPEG bytes for MJPEG streaming."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Cannot open video source: {source}")

        frames_done = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, _ = self.process_frame(frame)
                ok, buf = cv2.imencode(".jpg", annotated)
                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                    )
                frames_done += 1
                if max_frames is not None and frames_done >= max_frames:
                    break
        finally:
            cap.release()
            self.tracker.reset()

    def process_video_file(
        self,
        input_path: str,
        output_path: str,
    ) -> dict[str, Any]:
        """Write annotated video and return summary statistics."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open uploaded video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.tracker.reset()
        frame_count = 0
        total_ms = 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.perf_counter()
                annotated, _ = self.process_frame(frame)
                total_ms += (time.perf_counter() - t0) * 1000.0
                writer.write(annotated)
                frame_count += 1
        finally:
            cap.release()
            writer.release()
            self.tracker.reset()

        avg_fps = frame_count / (total_ms / 1000.0) if total_ms > 0 else 0.0
        return {
            "frames_processed": frame_count,
            "avg_fps": round(avg_fps, 2),
            "output_path": output_path,
        }


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    model: str
    mAP50: float
    device: str


class MetricsResponse(BaseModel):
    total_requests: int
    avg_inference_ms: float
    uptime_seconds: float


class FrameRequest(BaseModel):
    """Base64-encoded image (JPEG/PNG) for JSON inference."""

    image_b64: str = Field(..., description="Base64-encoded image file bytes")
    enable_lanes: bool = True
    return_image: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_detector()
    _get_tracker()
    _get_lane_detector()
    yield


app = FastAPI(
    title="ADAS Multi-Object Tracking API",
    description="YOLOv8 detection, DeepSORT tracking, and OpenCV lane detection",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=_model_name(),
        mAP50=_model_map50(),
        device=_device_label(),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    avg_ms = _total_inference_ms / _total_requests if _total_requests else 0.0
    return MetricsResponse(
        total_requests=_total_requests,
        avg_inference_ms=round(avg_ms, 3),
        uptime_seconds=round(time.time() - _app_start_time, 2),
    )


@app.post("/inference/image")
async def inference_image(
    file: UploadFile = File(...),
    enable_lanes: bool = True,
) -> JSONResponse:
    """Detect, track, and optionally find lanes on an uploaded image."""
    data = await file.read()
    frame = _decode_upload_to_bgr(data)
    pipeline = VideoPipeline(enable_lanes=enable_lanes)
    t0 = time.perf_counter()
    annotated, meta = pipeline.process_frame(frame)
    _record_inference((time.perf_counter() - t0) * 1000.0)
    meta["image_b64"] = _encode_image_b64(annotated)
    return JSONResponse(content=meta)


@app.post("/inference/frame")
async def inference_frame(body: FrameRequest) -> JSONResponse:
    """JSON endpoint: base64 image in, tracks + optional annotated image out."""
    try:
        raw = base64.b64decode(body.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64") from exc

    frame = _decode_upload_to_bgr(raw)
    pipeline = VideoPipeline(enable_lanes=body.enable_lanes)
    t0 = time.perf_counter()
    annotated, meta = pipeline.process_frame(frame)
    _record_inference((time.perf_counter() - t0) * 1000.0)
    if body.return_image:
        meta["image_b64"] = _encode_image_b64(annotated)
    return JSONResponse(content=meta)


@app.post("/inference/video")
async def inference_video(file: UploadFile = File(...)) -> JSONResponse:
    """Process an uploaded video; returns path to temp output and stats."""
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as inp:
        inp.write(await file.read())
        input_path = inp.name

    output_path = input_path + "_out.mp4"
    try:
        pipeline = VideoPipeline()
        t0 = time.perf_counter()
        stats = pipeline.process_video_file(input_path, output_path)
        _record_inference((time.perf_counter() - t0) * 1000.0)
        with open(output_path, "rb") as f:
            stats["video_b64"] = base64.b64encode(f.read()).decode("ascii")
        return JSONResponse(content=stats)
    finally:
        for path in (input_path, output_path):
            if os.path.isfile(path):
                os.remove(path)


@app.get("/stream/camera")
def stream_camera(
    device_id: int = 0,
    max_frames: int | None = None,
) -> StreamingResponse:
    """MJPEG stream from a local camera index (default 0)."""
    pipeline = VideoPipeline()

    def generate():
        try:
            yield from pipeline.run_on_capture(device_id, max_frames=max_frames)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/video")
def stream_video_file(
    path: str,
    max_frames: int | None = None,
) -> StreamingResponse:
    """
    MJPEG stream from a video file on disk.

    Path must be under the project data/ directory for safety.
    """
    data_dir = os.path.join(_project_root(), "data")
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(os.path.abspath(data_dir)):
        raise HTTPException(status_code=403, detail="Path must be under data/")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Video not found")

    pipeline = VideoPipeline()
    return StreamingResponse(
        pipeline.run_on_capture(abs_path, max_frames=max_frames),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
