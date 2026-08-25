"""Smooth Optical Aperture Segmenter with Chromatic Fringe Preservation.

Fits physically grounded rotated ellipses to hand-held optical apertures by combining
radial chroma-gradient analysis with robust RANSAC conic estimation, completely eliminating
local edge jutting, jagged bumps, and sensor noise artifacts.
"""

from __future__ import annotations

import cv2
import numpy as np
from portpan.core.exceptions import ApertureDetectionError
from portpan.core.models import CircleAperture
from portpan.segmentation.base import BaseApertureSegmenter


class PolarChromaSegmenter(BaseApertureSegmenter):
    """Detects optical apertures with silky-smooth, physically authentic rotated ellipse boundaries."""

    def __init__(
        self,
        num_angles: int = 360,
        min_radius_ratio: float = 0.25,
        max_radius_ratio: float = 0.65,
        black_floor_threshold: float = 28.0,
        trim_chroma_factor: float = 0.95,
    ) -> None:
        self.num_angles = num_angles
        self.min_radius_ratio = min_radius_ratio
        self.max_radius_ratio = max_radius_ratio
        self.black_floor_threshold = black_floor_threshold
        self.trim_chroma_factor = trim_chroma_factor

    def segment(self, image: np.ndarray) -> CircleAperture:
        """Detect the optical aperture boundary, fitting a smooth rotated ellipse to the chromatic fringe outer rim."""
        if image is None or image.size == 0:
            raise ApertureDetectionError("Image is empty or None")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        b, g, r = (
            cv2.split(image.astype(np.float32))
            if len(image.shape) == 3
            else (gray.astype(np.float32), gray.astype(np.float32), gray.astype(np.float32))
        )

        # Compute chromatic divergence map: high inside chromatic aberration / dispersion rings
        chroma = np.abs(b - r) + np.abs(g - r)

        # Work on downsampled resolution for fast, robust processing
        target_dim = 800
        scale = target_dim / max(w, h) if max(w, h) > target_dim else 1.0
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray.copy()
        small_chroma = cv2.resize(chroma, (int(w * scale), int(h * scale))) if scale < 1.0 else chroma.copy()
        sh, sw = small_gray.shape[:2]

        blurred = cv2.GaussianBlur(small_gray, (15, 15), 3.0)
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 1. Coarse Candidate Contour Search
        best_cnt: np.ndarray | None = None
        best_score = -1.0
        thresh_candidates = [
            int(otsu_val * 0.5),
            int(otsu_val * 0.7),
            int(otsu_val),
            35,
            45,
            55,
            65,
        ]

        min_r_scaled = min(sw, sh) * self.min_radius_ratio
        max_r_scaled = min(sw, sh) * self.max_radius_ratio

        for th in thresh_candidates:
            _, binary = cv2.threshold(blurred, th, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw >= sw * 0.95 and bh >= sh * 0.95:
                    continue
                (cx, cy), cr = cv2.minEnclosingCircle(c)
                if min_r_scaled <= cr <= max_r_scaled:
                    area = cv2.contourArea(c)
                    score = area / (np.pi * cr * cr + 1e-5)
                    if score > best_score:
                        best_score = score
                        best_cnt = c

        if best_cnt is not None and len(best_cnt) >= 5:
            (ecx, ecy), (eax1, eax2), e_angle = cv2.fitEllipse(best_cnt)
        else:
            ecx, ecy = sw / 2.0, sh / 2.0
            eax1, eax2 = min(sw, sh) * 0.92, min(sw, sh) * 0.92
            e_angle = 0.0

        # 2. Radial Ray Sampling for Chromatic Outer Rim Points
        angles = np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False)
        r_base = min(eax1, eax2) * 0.5

        edge_points: list[list[float]] = []
        for theta in angles:
            # Cast ray outward across the candidate transition band
            ray_r = np.linspace(r_base * 0.65, r_base * 1.35, 120)
            rx = ecx + ray_r * np.cos(theta)
            ry = ecy + ray_r * np.sin(theta)

            valid = (rx >= 0) & (rx < sw) & (ry >= 0) & (ry < sh)
            if not np.any(valid):
                continue

            rx_v = rx[valid].astype(np.float32)
            ry_v = ry[valid].astype(np.float32)

            # Sample brightness and chroma along ray
            vals = cv2.remap(small_gray.astype(np.float32), rx_v[:, None], ry_v[:, None], cv2.INTER_LINEAR).ravel()
            chroma_vals = cv2.remap(small_chroma.astype(np.float32), rx_v[:, None], ry_v[:, None], cv2.INTER_LINEAR).ravel()

            # Compute radial derivative
            grad = np.zeros_like(vals)
            grad[1:-1] = vals[:-2] - vals[2:]

            # Score transition: steep drop + chromatic dispersion peaking right at black floor
            score = np.clip(grad, 0, 100) * 1.6 + chroma_vals * 0.4 - np.abs(vals - self.black_floor_threshold) * 1.2
            best_idx = int(np.argmax(score))

            if vals[best_idx] < 60:
                edge_points.append([float(rx_v[best_idx]), float(ry_v[best_idx])])

        edge_pts_np = np.array(edge_points, dtype=np.float32)

        # 3. Robust RANSAC Conic / Ellipse Fitting
        final_ellipse = ((ecx, ecy), (eax1, eax2), e_angle)
        if len(edge_pts_np) >= 20:
            best_inliers: np.ndarray = np.array([], dtype=np.int32)
            best_candidate_ellipse = None

            num_samples = len(edge_pts_np)
            for _ in range(50):
                idx = np.random.choice(num_samples, 6, replace=False)
                sample_pts = edge_pts_np[idx]
                try:
                    ell = cv2.fitEllipseDirect(sample_pts.astype(np.int32))
                    (ccx, ccy), (cax1, cax2), c_ang = ell
                    if not (min(sw, sh) * 0.5 <= min(cax1, cax2) <= min(sw, sh) * 1.3):
                        continue

                    # Evaluate algebraic distance
                    rad_ang = np.deg2rad(c_ang)
                    cos_a, sin_a = np.cos(rad_ang), np.sin(rad_ang)
                    dx = edge_pts_np[:, 0] - ccx
                    dy = edge_pts_np[:, 1] - ccy
                    x_rot = dx * cos_a + dy * sin_a
                    y_rot = -dx * sin_a + dy * cos_a
                    a_semi, b_semi = cax1 / 2.0, cax2 / 2.0
                    dist = np.abs((x_rot / (a_semi + 1e-5)) ** 2 + (y_rot / (b_semi + 1e-5)) ** 2 - 1.0)
                    inliers = np.where(dist < 0.08)[0]
                    if len(inliers) > len(best_inliers):
                        best_inliers = inliers
                        best_candidate_ellipse = ell
                except Exception:
                    continue

            if best_candidate_ellipse is not None and len(best_inliers) >= 30:
                inlier_pts = edge_pts_np[best_inliers].astype(np.int32)
                final_ellipse = cv2.fitEllipseDirect(inlier_pts)
            else:
                final_ellipse = cv2.fitEllipse(edge_pts_np.astype(np.int32))

        (fcx, fcy), (fax1, fax2), f_angle = final_ellipse

        # Scale to full resolution with chroma trim factor to remove dark barrel bevel
        full_cx = float(fcx / scale)
        full_cy = float(fcy / scale)
        full_ax1 = float((fax1 / scale) * self.trim_chroma_factor)
        full_ax2 = float((fax2 / scale) * self.trim_chroma_factor)
        full_r = float((full_ax1 + full_ax2) / 4.0)

        # 4. Generate Parametrically Smooth Polygon (C^infinity continuous, zero jutting)
        poly_pts: list[list[float]] = []
        rad_ang = np.deg2rad(f_angle)
        cos_a, sin_a = np.cos(rad_ang), np.sin(rad_ang)
        a_semi, b_semi = full_ax1 / 2.0, full_ax2 / 2.0
        for t in np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False):
            px = full_cx + a_semi * np.cos(t) * cos_a - b_semi * np.sin(t) * sin_a
            py = full_cy + a_semi * np.cos(t) * sin_a + b_semi * np.sin(t) * cos_a
            poly_pts.append([float(px), float(py)])

        return CircleAperture(
            center_x=full_cx,
            center_y=full_cy,
            radius=full_r,
            image_width=w,
            image_height=h,
            boundary_polygon=poly_pts,
            ellipse_axes=(full_ax1, full_ax2),
            ellipse_angle=float(f_angle),
        )
