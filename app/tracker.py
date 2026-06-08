"""DeepSORT multi-object tracking on top of detector outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.detector import Detection


@dataclass(frozen=True)
class Track:
    """Confirmed or tentative track with axis-aligned bbox (ltrb)."""

    track_id: int
    bbox: tuple[float, float, float, float]  # left, top, right, bottom
    class_name: str
    confidence: float
    is_confirmed: bool


class ObjectTracker:
    """DeepSORT tracker using deep-sort-realtime."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        embedder_gpu: bool = False,
    ) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=embedder_gpu,
        )

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[Track]:
        """
        Associate detections with tracks for the current frame.

        DeepSORT expects detections as (ltwh, confidence, class_name).
        """
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            ltwh = [x1, y1, w, h]
            raw_dets.append((ltwh, det.confidence, det.class_name))

        tracks = self._tracker.update_tracks(raw_dets, frame=frame)
        output: list[Track] = []

        for track in tracks:
            if not track.is_confirmed() and track.time_since_update > 0:
                continue
            ltrb = track.to_ltrb(orig=True)
            if ltrb is None:
                ltrb = track.to_ltrb()
            bbox = tuple(float(v) for v in ltrb)
            det_class = track.get_det_class() or "unknown"
            conf = float(track.det_conf) if track.det_conf is not None else 0.0
            output.append(
                Track(
                    track_id=int(track.track_id),
                    bbox=bbox,
                    class_name=str(det_class),
                    confidence=conf,
                    is_confirmed=track.is_confirmed(),
                )
            )
        return output

    def reset(self) -> None:
        """Clear all track state (e.g. new video sequence)."""
        self._tracker.delete_all_tracks()
