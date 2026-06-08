"""Unit tests for YOLOv8 detector (mocked to avoid GPU/weight downloads in CI)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.detector import (  # noqa: E402
    COCO_TO_KITTI,
    Detection,
    ObjectDetector,
)


def _install_fake_ultralytics(mock_yolo_cls: MagicMock) -> None:
    fake = MagicMock()
    fake.YOLO = mock_yolo_cls
    sys.modules["ultralytics"] = fake


def _fake_boxes():
    """Simulate ultralytics Boxes tensors for two KITTI-relevant classes."""
    import torch

    xyxy = torch.tensor([[10.0, 20.0, 110.0, 220.0], [300.0, 50.0, 400.0, 200.0]])
    conf = torch.tensor([0.91, 0.82])
    cls = torch.tensor([0.0, 2.0])  # person, car
    boxes = MagicMock()
    boxes.xyxy = xyxy
    boxes.conf = conf
    boxes.cls = cls
    boxes.__len__ = lambda self: 2
    return boxes


def _fake_result():
    result = MagicMock()
    result.boxes = _fake_boxes()
    result.names = {0: "person", 2: "car"}
    return result


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((375, 1242, 3), dtype=np.uint8)


def test_detect_filters_kitti_classes(blank_frame):
    model = MagicMock()
    model.predict.return_value = [_fake_result()]
    mock_yolo_cls = MagicMock(return_value=model)
    _install_fake_ultralytics(mock_yolo_cls)

    detector = ObjectDetector(model_path="yolov8n.pt")
    detections = detector.detect(blank_frame)

    assert len(detections) == 2
    assert detections[0].class_name == "pedestrian"
    assert detections[1].class_name == "car"
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].bbox == (10.0, 20.0, 110.0, 220.0)


def test_detect_empty_boxes(blank_frame):
    model = MagicMock()
    empty = MagicMock()
    empty.boxes = None
    model.predict.return_value = [empty]
    mock_yolo_cls = MagicMock(return_value=model)
    _install_fake_ultralytics(mock_yolo_cls)

    detector = ObjectDetector()
    assert detector.detect(blank_frame) == []


def test_coco_to_kitti_mapping_covers_vehicle_classes():
    mock_yolo_cls = MagicMock()
    _install_fake_ultralytics(mock_yolo_cls)
    detector = ObjectDetector()
    assert "car" in detector.class_names
    assert COCO_TO_KITTI[2] == "car"
    assert COCO_TO_KITTI[0] == "pedestrian"
    assert COCO_TO_KITTI[1] == "cyclist"


def test_detection_dataclass():
    det = Detection(
        bbox=(0.0, 1.0, 2.0, 3.0),
        confidence=0.5,
        class_name="car",
        class_id=2,
    )
    assert det.class_name == "car"
