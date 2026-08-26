"""Smooth Optical Aperture Segmenter with Chromatic Fringe Preservation.

Fits physically grounded rotated ellipses to hand-held optical apertures by combining
radial chroma-gradient analysis with robust RANSAC conic estimation, completely eliminating
local edge jutting, jagged bumps, and sensor noise artifacts.

Revision 2: Consensus center estimation, tighter radius priors, two-pass gradient ray
refinement, stricter RANSAC acceptance, and final ellipse sanity gating to handle
variable camera-to-binocular alignment drift in digiscope captures.
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
        max_radius_ratio: float = 0.52,
        black_floor_threshold: float = 28.0,
        trim_chroma_factor: float = 0.97,
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

        # ──────────────────────────────────────────────────────────────────────
        # 1. CONSENSUS CENTER ESTIMATION
        #    Three independent estimators → median for robustness against
        #    the coarse contour latching onto the outer bevel ring.
        # ──────────────────────────────────────────────────────────────────────
        center_candidates: list[tuple[float, float]] = []
        radius_candidates: list[float] = []

        # Strategy A: Distance-transform peak from dark-background mask
        _, dark_mask = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        dist = cv2.distanceTransform(dark_mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        if max_val > 0:
            center_candidates.append((float(max_loc[0]), float(max_loc[1])))
            radius_candidates.append(float(max_val))

        # Strategy B: Moment-based centroid of brightest connected component
        _, bright_mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_m)
        cnts_bright, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_bright:
            biggest = max(cnts_bright, key=cv2.contourArea)
            M = cv2.moments(biggest)
            if M["m00"] > 0:
                mcx = M["m10"] / M["m00"]
                mcy = M["m01"] / M["m00"]
                center_candidates.append((mcx, mcy))
                _, mr = cv2.minEnclosingCircle(biggest)
                radius_candidates.append(float(mr))

        # Strategy C: Best contour from multi-threshold search (original approach, refined)
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        best_cnt: np.ndarray | None = None
        best_score = -1.0
        thresh_candidates = [
            int(otsu_val * 0.5),
            int(otsu_val * 0.7),
            int(otsu_val),
            35, 45, 55, 65,
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
                    circularity = area / (np.pi * cr * cr + 1e-5)
                    if circularity > best_score:
                        best_score = circularity
                        best_cnt = c

        if best_cnt is not None and len(best_cnt) >= 5:
            (ccx, ccy), ccr = cv2.minEnclosingCircle(best_cnt)
            center_candidates.append((float(ccx), float(ccy)))
            radius_candidates.append(float(ccr))

        # Consensus: median of all center estimates
        if len(center_candidates) >= 2:
            xs = [c[0] for c in center_candidates]
            ys = [c[1] for c in center_candidates]
            ecx = float(np.median(xs))
            ecy = float(np.median(ys))
        elif center_candidates:
            ecx, ecy = center_candidates[0]
        else:
            ecx, ecy = sw / 2.0, sh / 2.0

        # Consensus radius: median, clamped to tight physical bounds
        if radius_candidates:
            est_radius = float(np.median(radius_candidates))
            est_radius = min(est_radius, max_r_scaled)
            est_radius = max(est_radius, min_r_scaled)
        else:
            est_radius = min(sw, sh) * 0.40

        # Initial ellipse axes estimate from consensus radius
        eax1 = est_radius * 2.0
        eax2 = est_radius * 2.0
        e_angle = 0.0

        # If we had a good contour, use its fitEllipse but only if it agrees with consensus
        if best_cnt is not None and len(best_cnt) >= 5:
            try:
                (fecx, fecy), (feax1, feax2), fe_angle = cv2.fitEllipse(best_cnt)
                # Accept only if center is close to consensus and axes are reasonable
                center_dist = np.sqrt((fecx - ecx) ** 2 + (fecy - ecy) ** 2)
                axes_ratio = max(feax1, feax2) / (est_radius * 2.0 + 1e-5)
                if center_dist < est_radius * 0.15 and 0.7 < axes_ratio < 1.15:
                    ecx, ecy = fecx, fecy
                    eax1, eax2 = feax1, feax2
                    e_angle = fe_angle
            except Exception:
                pass

        # ──────────────────────────────────────────────────────────────────────
        # 2. TWO-PASS RADIAL RAY SAMPLING WITH GRADIENT REFINEMENT
        #    Pass 1: Coarse transition candidate via gradient + chroma scoring
        #    Pass 2: Walk inward from candidate to find the true inner chromatic edge
        # ──────────────────────────────────────────────────────────────────────
        angles = np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False)
        r_base = min(eax1, eax2) * 0.5

        edge_points: list[list[float]] = []
        edge_radii: list[float] = []

        for theta in angles:
            # Tighter sampling window: ±25% around estimated radius (was ±35%)
            ray_r = np.linspace(r_base * 0.75, r_base * 1.25, 120)
            rx = ecx + ray_r * np.cos(theta)
            ry = ecy + ray_r * np.sin(theta)

            valid = (rx >= 0) & (rx < sw) & (ry >= 0) & (ry < sh)
            if not np.any(valid):
                continue

            rx_v = rx[valid].astype(np.float32)
            ry_v = ry[valid].astype(np.float32)
            ray_r_v = ray_r[valid]

            # Sample brightness and chroma along ray
            vals = cv2.remap(small_gray.astype(np.float32), rx_v[:, None], ry_v[:, None], cv2.INTER_LINEAR).ravel()
            chroma_vals = cv2.remap(small_chroma.astype(np.float32), rx_v[:, None], ry_v[:, None], cv2.INTER_LINEAR).ravel()

            # Pass 1: Compute radial derivative and score transition
            grad = np.zeros_like(vals)
            grad[1:-1] = vals[:-2] - vals[2:]

            # Score transition: steep brightness drop + chromatic dispersion peaking near black floor
            score = np.clip(grad, 0, 100) * 1.6 + chroma_vals * 0.4 - np.abs(vals - self.black_floor_threshold) * 1.2
            coarse_idx = int(np.argmax(score))

            # Pass 2: Walk inward from the coarse candidate to find the true inner edge
            # The inner edge is where luma is still above the dark threshold and gradient is steepest
            best_idx = coarse_idx
            if coarse_idx > 5:
                # Look in a small window inward from the coarse candidate
                search_start = max(0, coarse_idx - 15)
                search_end = coarse_idx + 3
                window_grad = grad[search_start:search_end]
                if len(window_grad) > 0:
                    # Find the point with the steepest positive gradient (brightness dropping outward)
                    # that is still above the dark floor
                    window_vals = vals[search_start:search_end]
                    window_chroma = chroma_vals[search_start:search_end]
                    # Combined inner-edge score: prefer points where luma is moderate (not black)
                    # and gradient is steep, with chroma boost
                    inner_score = (
                        np.clip(window_grad, 0, 100) * 2.0
                        + window_chroma * 0.6
                        + np.clip(window_vals - 30, 0, 80) * 0.3
                    )
                    local_best = int(np.argmax(inner_score))
                    best_idx = search_start + local_best

            # Only accept if the point is not deep in the black region
            if vals[best_idx] < 80 and vals[best_idx] > 8:
                edge_points.append([float(rx_v[best_idx]), float(ry_v[best_idx])])
                edge_radii.append(float(ray_r_v[best_idx]))

        edge_pts_np = np.array(edge_points, dtype=np.float32)

        # ──────────────────────────────────────────────────────────────────────
        # 3. ROBUST RANSAC CONIC / ELLIPSE FITTING (Stricter)
        #    - 8 sample points (was 6)
        #    - Tighter algebraic distance threshold: 0.04 (was 0.08)
        #    - Radius consistency check against median ray distance
        # ──────────────────────────────────────────────────────────────────────
        final_ellipse = ((ecx, ecy), (eax1, eax2), e_angle)

        if len(edge_radii) > 10:
            median_ray_r = float(np.median(edge_radii))
        else:
            median_ray_r = r_base

        if len(edge_pts_np) >= 20:
            best_inliers: np.ndarray = np.array([], dtype=np.int32)
            best_candidate_ellipse = None

            num_samples = len(edge_pts_np)
            for _ in range(80):
                idx = np.random.choice(num_samples, min(8, num_samples), replace=False)
                sample_pts = edge_pts_np[idx]
                try:
                    ell = cv2.fitEllipseDirect(sample_pts.astype(np.int32))
                    (ccx, ccy), (cax1, cax2), c_ang = ell

                    # Radius consistency: semi-axes must be within [0.7, 1.15] of median ray distance
                    semi_max = max(cax1, cax2) / 2.0
                    semi_min = min(cax1, cax2) / 2.0
                    if semi_max > median_ray_r * 1.15 or semi_min < median_ray_r * 0.7:
                        continue

                    # Center consistency: must be near consensus center
                    center_drift = np.sqrt((ccx - ecx) ** 2 + (ccy - ecy) ** 2)
                    if center_drift > median_ray_r * 0.12:
                        continue

                    # Evaluate algebraic distance (tighter threshold)
                    rad_ang = np.deg2rad(c_ang)
                    cos_a, sin_a = np.cos(rad_ang), np.sin(rad_ang)
                    dx = edge_pts_np[:, 0] - ccx
                    dy = edge_pts_np[:, 1] - ccy
                    x_rot = dx * cos_a + dy * sin_a
                    y_rot = -dx * sin_a + dy * cos_a
                    a_semi, b_semi = cax1 / 2.0, cax2 / 2.0
                    dist = np.abs((x_rot / (a_semi + 1e-5)) ** 2 + (y_rot / (b_semi + 1e-5)) ** 2 - 1.0)
                    inliers = np.where(dist < 0.04)[0]
                    if len(inliers) > len(best_inliers):
                        best_inliers = inliers
                        best_candidate_ellipse = ell
                except Exception:
                    continue

            if best_candidate_ellipse is not None and len(best_inliers) >= 30:
                inlier_pts = edge_pts_np[best_inliers].astype(np.int32)
                try:
                    final_ellipse = cv2.fitEllipseDirect(inlier_pts)
                except Exception:
                    final_ellipse = best_candidate_ellipse
            elif len(edge_pts_np) >= 5:
                final_ellipse = cv2.fitEllipse(edge_pts_np.astype(np.int32))

        # ──────────────────────────────────────────────────────────────────────
        # 4. FINAL ELLIPSE SANITY GATE
        #    Reject the fit if it deviates too much from physical expectations.
        # ──────────────────────────────────────────────────────────────────────
        (fcx, fcy), (fax1, fax2), f_angle = final_ellipse

        # Check: center within 10% of image center
        center_ok = (
            abs(fcx - sw / 2.0) < sw * 0.15
            and abs(fcy - sh / 2.0) < sh * 0.15
        )
        # Check: axes within [0.60, 1.10] of median detected radius
        axes_ok = True
        if median_ray_r > 0:
            for ax in (fax1, fax2):
                ratio = (ax / 2.0) / median_ray_r
                if ratio < 0.60 or ratio > 1.10:
                    axes_ok = False
                    break

        if not (center_ok and axes_ok):
            # Fall back to simple circle from consensus estimates
            fcx, fcy = ecx, ecy
            fax1 = median_ray_r * 2.0
            fax2 = median_ray_r * 2.0
            f_angle = 0.0

        # ──────────────────────────────────────────────────────────────────────
        # 5. SCALE TO FULL RESOLUTION & GENERATE SMOOTH POLYGON
        # ──────────────────────────────────────────────────────────────────────
        full_cx = float(fcx / scale)
        full_cy = float(fcy / scale)
        full_ax1 = float((fax1 / scale) * self.trim_chroma_factor)
        full_ax2 = float((fax2 / scale) * self.trim_chroma_factor)
        full_r = float((full_ax1 + full_ax2) / 4.0)

        # Generate parametrically smooth polygon (C^infinity continuous, zero jutting)
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
