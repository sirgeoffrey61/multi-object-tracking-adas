"""YOLOv8 object detection filtered for KITTI-relevant ADAS classes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

# COCO class IDs used by YOLOv8 for cars, pedestrians, and cyclists.
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7

KITTI_CLASS_NAMES = ("car", "pedestrian", "cyclist")

# Map COCO indices to unified KITTI-style labels.
COCO_TO_KITTI: dict[int, str] = {
    COCO_PERSON: "pedestrian",
    COCO_BICYCLE: "cyclist",
    COCO_CAR: "car",
    COCO_MOTORCYCLE: "cyclist",
    COCO_BUS: "car",
    COCO_TRUCK: "car",
}

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.45


@dataclass(frozen=True)
class Detection:
    """Single detection in pixel coordinates (xyxy)."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    class_id: int


class ObjectDetector:
    """YOLOv8 wrapper with KITTI-oriented class filtering."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_weights = os.path.join(root, "models", "yolov8n.pt")
        weights = model_path or default_weights

        if not os.path.isfile(weights):
            # Ultralytics downloads yolov8n.pt on first use when given a hub name.
            weights = "yolov8n.pt"

        self.model = YOLO(weights)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.class_names = KITTI_CLASS_NAMES

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR numpy image and return filtered detections."""
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(frame, **kwargs)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            if cls_id not in COCO_TO_KITTI:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_name=COCO_TO_KITTI[cls_id],
                    class_id=cls_id,
                )
            )
        return detections

    def detect_raw(self, frame: np.ndarray) -> list[Detection]:
        """Return all COCO detections without KITTI class filtering (debug/eval)."""
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(frame, **kwargs)
        if not results or results[0].boxes is None:
            return []

        result = results[0]
        boxes = result.boxes
        names = result.names
        detections: list[Detection] = []

        for box, conf, cls_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = (float(v) for v in box)
            label = names.get(int(cls_id), str(cls_id))
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_name=label,
                    class_id=int(cls_id),
                )
            )
        return detections
