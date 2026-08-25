"""Polar Chroma-Aware Aperture Segmenter.

Unrolls the optical boundary into polar coordinate space, detects the outer transition
where chromatic aberration fringes meet the black sensor background, and fits a smooth,
sub-pixel continuous boundary polygon.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from portpan.core.exceptions import ApertureDetectionError
from portpan.core.models import CircleAperture
from portpan.segmentation.base import BaseApertureSegmenter


class PolarChromaSegmenter(BaseApertureSegmenter):
    """Detects hand-held optical apertures by analyzing polar radial chroma dispersion and luminance falloff."""

    def __init__(
        self,
        num_angles: int = 360,
        radial_samples: int = 300,
        min_radius_ratio: float = 0.25,
        max_radius_ratio: float = 0.65,
        black_floor_threshold: float = 30.0,
    ) -> None:
        self.num_angles = num_angles
        self.radial_samples = radial_samples
        self.min_radius_ratio = min_radius_ratio
        self.max_radius_ratio = max_radius_ratio
        self.black_floor_threshold = black_floor_threshold

    def segment(self, image: np.ndarray) -> CircleAperture:
        """Detect the optical aperture with sub-pixel polygon hugging the chromatic fringe outer boundary."""
        if image is None or image.size == 0:
            raise ApertureDetectionError("Image is empty or None")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        b, g, r = (
            cv2.split(image.astype(np.float32))
            if len(image.shape) == 3
            else (gray.astype(np.float32), gray.astype(np.float32), gray.astype(np.float32))
        )

        # Compute chromatic divergence map: high inside dispersion rings
        chroma = np.abs(b - r) + np.abs(g - r)

        # Work on downsampled resolution for fast, robust contour detection
        target_dim = 800
        scale = target_dim / max(w, h) if max(w, h) > target_dim else 1.0
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray.copy()
        small_chroma = cv2.resize(chroma, (int(w * scale), int(h * scale))) if scale < 1.0 else chroma.copy()
        sh, sw = small_gray.shape[:2]

        blurred = cv2.GaussianBlur(small_gray, (11, 11), 2.0)

        # Compute Otsu baseline
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find initial coarse candidate contour
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
                    ]
                )
            )
        )

        min_r_scaled = min(sw, sh) * self.min_radius_ratio
        max_r_scaled = min(sw, sh) * self.max_radius_ratio

        best_contour: np.ndarray | None = None
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

                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw >= sw * 0.96 and bh >= sh * 0.96:
                    continue

                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
                (scx, scy), sr = cv2.minEnclosingCircle(cnt)

                if min_r_scaled <= sr <= max_r_scaled:
                    center_dist = np.hypot(scx - sw / 2.0, scy - sh / 2.0) / max(sw, sh)
                    score = circularity * (1.0 - 0.25 * center_dist)
                    if score > best_score:
                        best_score = score
                        best_contour = cnt

        if best_contour is None or len(best_contour) < 5:
            # Fallback to centered default
            cx, cy = w / 2.0, h / 2.0
            r_mean = min(w, h) * 0.46
            return CircleAperture(
                center_x=cx,
                center_y=cy,
                radius=r_mean,
                image_width=w,
                image_height=h,
            )

        # Fit initial rotated ellipse to candidate contour
        (scx, scy), (sax1, sax2), s_angle = cv2.fitEllipse(best_contour)

        # Unroll the boundary region in polar space around (scx, scy)
        r_search_min = min(sax1, sax2) * 0.35
        r_search_max = max(sax1, sax2) * 0.70

        angles = np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False)
        radii = np.linspace(r_search_min, r_search_max, self.radial_samples)

        grid_angles, grid_radii = np.meshgrid(angles, radii)
        map_x = (scx + grid_radii * np.cos(grid_angles)).astype(np.float32)
        map_y = (scy + grid_radii * np.sin(grid_angles)).astype(np.float32)

        polar_gray = cv2.remap(small_gray.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        polar_chroma = cv2.remap(small_chroma.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)

        # Compute radial brightness derivative (positive when brightness drops outward)
        grad_r = np.zeros_like(polar_gray)
        grad_r[1:-1, :] = polar_gray[:-2, :] - polar_gray[2:, :]

        # Edge cost function: peak where brightness drops sharply, chromatic fringe is present, and intensity reaches black floor
        cost_map = np.zeros_like(polar_gray)
        for r_idx in range(self.radial_samples):
            intensity = polar_gray[r_idx, :]
            g_val = np.clip(grad_r[r_idx, :], 0, 100)
            c_val = polar_chroma[r_idx, :]
            # Reward steep gradient and chromatic fringe right at black boundary (< 35)
            cost_map[r_idx, :] = (
                g_val * 1.6
                + c_val * 0.5
                - np.abs(intensity - self.black_floor_threshold) * 1.2
            )

        # Extract optimal radial boundary per angle
        optimal_r_indices = np.argmax(cost_map, axis=0)

        # Smooth radial curve using circular-wrapping median and Gaussian filters
        raw_radii = radii[optimal_r_indices]
        smooth_radii = gaussian_filter1d(
            median_filter(raw_radii, size=15, mode="wrap"), sigma=3.5, mode="wrap"
        )

        # Map back to full-resolution Cartesian coordinates
        poly_x = (scx + smooth_radii * np.cos(angles)) / scale
        poly_y = (scy + smooth_radii * np.sin(angles)) / scale

        # Boundary polygon in full image pixel coordinates
        boundary_polygon = np.column_stack([poly_x, poly_y]).tolist()

        # Mean radius and center in full-resolution image space
        orig_cx = float(scx / scale)
        orig_cy = float(scy / scale)
        orig_mean_r = float(np.mean(smooth_radii) / scale)

        return CircleAperture(
            center_x=orig_cx,
            center_y=orig_cy,
            radius=orig_mean_r,
            image_width=w,
            image_height=h,
            boundary_polygon=boundary_polygon,
            ellipse_axes=(float(sax1 / scale), float(sax2 / scale)),
            ellipse_angle=float(s_angle),
        )
