"""Classical lane detection using OpenCV (Canny + Hough + ROI)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneLines:
    """Left and right lane polylines as (x1, y1, x2, y2) segments."""

    left: tuple[int, int, int, int] | None
    right: tuple[int, int, int, int] | None


class LaneDetector:
    """Bird's-eye style lane finder for forward-facing dashcam / KITTI images."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 40,
        min_line_length: int = 40,
        max_line_gap: int = 150,
        roi_vertices: np.ndarray | None = None,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self._roi_vertices = roi_vertices

    def _region_of_interest(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if self._roi_vertices is not None:
            vertices = self._roi_vertices
        else:
            vertices = np.array(
                [
                    [
                        (int(width * 0.05), height),
                        (int(width * 0.45), int(height * 0.58)),
                        (int(width * 0.55), int(height * 0.58)),
                        (int(width * 0.95), height),
                    ]
                ],
                dtype=np.int32,
            )
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(image, mask)

    @staticmethod
    def _average_slope_intercept(
        lines: np.ndarray | None,
        image_shape: tuple[int, ...],
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        if lines is None:
            return None, None

        height, width = image_shape[:2]
        y1 = height
        y2 = int(height * 0.6)

        left_fit: list[float] = []
        right_fit: list[float] = []

        for line in lines:
            for x1, y1_l, x2, y2_l in line:
                if x2 == x1:
                    continue
                slope = (y2_l - y1_l) / (x2 - x1)
                intercept = y1_l - slope * x1
                if slope < -0.3:
                    left_fit.append((slope, intercept))
                elif slope > 0.3:
                    right_fit.append((slope, intercept))

        def _line_from_fits(fits: list[float]) -> tuple[int, int, int, int] | None:
            if not fits:
                return None
            slope = float(np.mean([f[0] for f in fits]))
            intercept = float(np.mean([f[1] for f in fits]))
            x1_out = int((y1 - intercept) / slope)
            x2_out = int((y2 - intercept) / slope)
            return (x1_out, y1, x2_out, y2)

        return _line_from_fits(left_fit), _line_from_fits(right_fit)

    def detect(self, frame: np.ndarray) -> LaneLines:
        """Detect lane lines on a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        masked = self._region_of_interest(edges)
        lines = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        left, right = self._average_slope_intercept(lines, frame.shape)
        return LaneLines(left=left, right=right)

    def draw_lanes(
        self,
        frame: np.ndarray,
        lanes: LaneLines,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 8,
    ) -> np.ndarray:
        """Overlay lane segments on a copy of the frame."""
        out = frame.copy()
        if lanes.left is not None:
            cv2.line(out, lanes.left[:2], lanes.left[2:], color, thickness)
        if lanes.right is not None:
            cv2.line(out, lanes.right[:2], lanes.right[2:], color, thickness)
        return out
