"""Robust circular aperture segmentation using adaptive thresholding, morphological contour analysis, and least-squares circle fitting."""

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
        max_radius_ratio: float = 0.65,
        dark_corner_threshold: float = 65.0,
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

        # Check corner vignetting
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

        # Downsample for fast and robust contour extraction
        target_dim = 800
        scale = target_dim / max(w, h) if max(w, h) > target_dim else 1.0
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray.copy()
        sh, sw = small_gray.shape[:2]

        blurred = cv2.GaussianBlur(small_gray, (11, 11), 2.0)

        # Compute Otsu baseline
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh_candidates = sorted(
            list(
                set(
                    [
                        max(10, int(otsu_val * 0.4)),
                        max(15, int(otsu_val * 0.6)),
                        max(20, int(otsu_val * 0.8)),
                        int(otsu_val),
                        30,
                        40,
                        50,
                        65,
                        80,
                    ]
                )
            )
        )

        min_r = min(w, h) * self.min_radius_ratio
        max_r = min(w, h) * self.max_radius_ratio

        best_circle: tuple[float, float, float] | None = None
        best_score = -1.0

        for thresh_val in thresh_candidates:
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0 or area <= 0:
                    continue

                # Reject contours that hug the full image frame border
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw >= sw * 0.96 and bh >= sh * 0.96:
                    continue

                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
                (scx, scy), sr = cv2.minEnclosingCircle(cnt)

                orig_cx = scx / scale
                orig_cy = scy / scale
                orig_r = sr / scale

                if min_r <= orig_r <= max_r:
                    # Prefer high circularity and plausible optical center proximity
                    center_dist = np.hypot(orig_cx - w / 2.0, orig_cy - h / 2.0) / max(w, h)
                    score = circularity * (1.0 - 0.25 * center_dist)

                    if score > best_score:
                        # Refine circle fit using least-squares if contour has enough points
                        if len(cnt) >= 10:
                            pts_orig = (cnt.reshape(-1, 2).astype(np.float64)) / scale
                            # Fast algebraic circle fit (Kåsa method)
                            x = pts_orig[:, 0]
                            y = pts_orig[:, 1]
                            A = np.column_stack([x, y, np.ones_like(x)])
                            b = x**2 + y**2
                            try:
                                c_fit, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                fit_cx = c_fit[0] / 2.0
                                fit_cy = c_fit[1] / 2.0
                                fit_r = np.sqrt(c_fit[2] + fit_cx**2 + fit_cy**2)
                                if min_r <= fit_r <= max_r and np.hypot(fit_cx - orig_cx, fit_cy - orig_cy) < orig_r * 0.2:
                                    orig_cx, orig_cy, orig_r = fit_cx, fit_cy, fit_r
                            except Exception:
                                pass

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

        # Strategy 2: Hough Circles Fallback
        dp = 1.2
        min_dist = int(sh * 0.3)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=50,
            param2=30,
            minRadius=int(sh * 0.25),
            maxRadius=int(sh * 0.65),
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

        # Fallback to centered inscribed optical circle
        return CircleAperture(
            center_x=w / 2.0,
            center_y=h / 2.0,
            radius=min(w, h) * 0.46,
            image_width=w,
            image_height=h,
        )
