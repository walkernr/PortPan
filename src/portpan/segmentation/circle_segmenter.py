"""Robust circular aperture segmentation using adaptive thresholding and contour fitting."""

from __future__ import annotations

import cv2
import numpy as np
from portpan.core.exceptions import ApertureDetectionError
from portpan.core.models import CircleAperture
from portpan.segmentation.base import BaseApertureSegmenter


class RobustCircleSegmenter(BaseApertureSegmenter):
    """Detects circular barrel aperture using multi-scale morphological contour analysis and Hough fallback."""

    def __init__(
        self,
        min_radius_ratio: float = 0.25,
        max_radius_ratio: float = 1.60,
        dark_corner_threshold: float = 50.0,
    ) -> None:
        self.min_radius_ratio = min_radius_ratio
        self.max_radius_ratio = max_radius_ratio
        self.dark_corner_threshold = dark_corner_threshold

    def segment(self, image: np.ndarray) -> CircleAperture:
        """Detect the optical circular aperture within the given image."""
        if image is None or image.size == 0:
            raise ApertureDetectionError("Image is empty or None")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Check if corners are dark (indicating barrel vignetting)
        corner_w = max(5, w // 20)
        corner_h = max(5, h // 20)
        corner_means = [
            float(np.mean(gray[:corner_h, :corner_w])),
            float(np.mean(gray[:corner_h, -corner_w:])),
            float(np.mean(gray[-corner_h:, :corner_w])),
            float(np.mean(gray[-corner_h:, -corner_w:])),
        ]
        avg_corner = float(np.mean(corner_means))
        center_mean = float(np.mean(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]))

        # If corners are bright relative to center, the image might not have a vignette aperture
        if avg_corner > self.dark_corner_threshold and avg_corner > 0.8 * center_mean:
            # Fallback to full frame inscribed circle
            cx, cy = w / 2.0, h / 2.0
            r = min(w, h) / 2.0
            return CircleAperture(
                center_x=cx,
                center_y=cy,
                radius=r,
                image_width=w,
                image_height=h,
            )

        # Strategy 1: Multi-threshold contour search on downsampled image
        target_dim = 800
        scale = target_dim / max(w, h) if max(w, h) > target_dim else 1.0
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray.copy()
        sh, sw = small_gray.shape[:2]

        # Slight blur to remove sensor noise
        blurred = cv2.GaussianBlur(small_gray, (9, 9), 2.0)

        # Try multiple threshold values to find the cleanest aperture contour
        best_circle: tuple[float, float, float] | None = None
        best_score = -1.0

        for thresh_val in [15, 25, 40, 60]:
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            # Morphological close to bridge small gaps and fill pinhole reflections
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter <= 0 or area <= 0:
                continue

            # Circularity: 4 * pi * area / perimeter^2 (1.0 for perfect circle)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            (scx, scy), sr = cv2.minEnclosingCircle(largest)

            # Map back to original scale
            orig_cx = scx / scale
            orig_cy = scy / scale
            orig_r = sr / scale

            # Score candidates based on area, circularity, and proximity to image center
            min_r = min(w, h) * self.min_radius_ratio
            max_r = max(w, h) * self.max_radius_ratio

            if min_r <= orig_r <= max_r:
                center_dist = np.hypot(orig_cx - w / 2.0, orig_cy - h / 2.0) / max(w, h)
                score = circularity * (1.0 - 0.3 * center_dist)
                if score > best_score:
                    best_score = score
                    best_circle = (orig_cx, orig_cy, orig_r)

        if best_circle is not None:
            cx, cy, r = best_circle
            return CircleAperture(
                center_x=float(cx),
                center_y=float(cy),
                radius=float(r),
                image_width=w,
                image_height=h,
            )

        # Strategy 2: Hough Circles fallback
        dp = 1.2
        min_dist = int(sh * 0.4)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=50,
            param2=30,
            minRadius=int(sh * 0.2),
            maxRadius=int(sh * 0.8),
        )

        if circles is not None and len(circles) > 0:
            c = circles[0][0]
            cx, cy, r = c[0] / scale, c[1] / scale, c[2] / scale
            return CircleAperture(
                center_x=float(cx),
                center_y=float(cy),
                radius=float(r),
                image_width=w,
                image_height=h,
            )

        # Default fallback: centered circle inscribed in frame
        return CircleAperture(
            center_x=w / 2.0,
            center_y=h / 2.0,
            radius=min(w, h) * 0.45,
            image_width=w,
            image_height=h,
        )
