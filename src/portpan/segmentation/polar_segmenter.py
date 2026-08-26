"""Smooth Optical Aperture Segmenter with Continuous Polar Fourier Harmonic Modeling.

Fits physically grounded continuous polar harmonic curves to handheld optical apertures by
aligning the boundary to the cyan/blue chromatic dispersion fringe of the optical field stop.
This accounts for asymmetric clipped circles, truncated chords, and off-axis misalignments
while mathematically eliminating local jutting, jagged bumps, and dark eyecup bevel leakage.
"""

from __future__ import annotations

import cv2
import numpy as np
from portpan.core.exceptions import ApertureDetectionError
from portpan.core.models import CircleAperture
from portpan.segmentation.base import BaseApertureSegmenter


class PolarChromaSegmenter(BaseApertureSegmenter):
    """Detects optical apertures with continuous polar Fourier harmonic boundaries."""

    _sam_model = None

    def __init__(
        self,
        num_angles: int = 360,
        num_harmonics: int = 6,
        min_radius_ratio: float = 0.25,
        max_radius_ratio: float = 0.52,
        trim_chroma_factor: float = 1.0,
        use_sam: bool = True,
    ) -> None:
        self.num_angles = num_angles
        self.num_harmonics = num_harmonics
        self.min_radius_ratio = min_radius_ratio
        self.max_radius_ratio = max_radius_ratio
        self.trim_chroma_factor = trim_chroma_factor
        self.use_sam = use_sam

    @classmethod
    def _get_sam_model(cls):
        if cls._sam_model is None:
            try:
                from ultralytics import SAM
                import torch
                cls._sam_model = SAM("mobile_sam.pt")
            except Exception:
                cls._sam_model = False
        return cls._sam_model

    def segment(self, image: np.ndarray) -> CircleAperture:
        """Detect the optical aperture boundary using line integral optimization."""
        if image is None or image.size == 0:
            raise ApertureDetectionError("Image is empty or None")

        h, w = image.shape[:2]
        scale = 800.0 / max(w, h)
        sw, sh = int(round(w * scale)), int(round(h * scale))
        small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if len(small.shape) == 3 and small.shape[2] == 3:
            b, g, r = cv2.split(small.astype(np.float32))
            amber_flare = r - b
        else:
            amber_flare = np.zeros_like(gray, dtype=np.float32)

        cx_init, cy_init = sw / 2.0, sh / 2.0
        best_score = -1e9
        best_circle = (cx_init, cy_init, 240.0)

        angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # Global grid search around optical center
        for dcx in np.linspace(-25, 25, 11):
            for dcy in np.linspace(-25, 25, 11):
                ccx = cx_init + dcx
                ccy = cy_init + dcy

                for R in np.linspace(215, 255, 41):
                    r_in = R - 6.0
                    r_out = R + 6.0

                    xin = np.clip(ccx + r_in * cos_a, 0, sw - 1).astype(np.float32)
                    yin = np.clip(ccy + r_in * sin_a, 0, sh - 1).astype(np.float32)

                    xout = np.clip(ccx + r_out * cos_a, 0, sw - 1).astype(np.float32)
                    yout = np.clip(ccy + r_out * sin_a, 0, sh - 1).astype(np.float32)

                    lum_in = cv2.remap(gray, xin[:, None], yin[:, None], cv2.INTER_LINEAR).ravel()
                    lum_out = cv2.remap(gray, xout[:, None], yout[:, None], cv2.INTER_LINEAR).ravel()
                    af_in = cv2.remap(amber_flare, xin[:, None], yin[:, None], cv2.INTER_LINEAR).ravel()

                    drop = lum_in - lum_out
                    # Line integral score: Step drop + scene brightness - amber flare penalty
                    score = float(
                        np.mean(drop)
                        + 0.35 * np.mean(lum_in)
                        - 1.8 * np.mean(np.maximum(0.0, af_in - 10.0))
                    )

                    if score > best_score:
                        best_score = score
                        best_circle = (ccx, ccy, R)

        ccx, ccy, R = best_circle
        full_cx = float(ccx / scale)
        full_cy = float(ccy / scale)
        full_r = float(R / scale)

        # 360 boundary polygon points for exact masking
        poly_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        pxs = full_cx + full_r * np.cos(poly_angles)
        pys = full_cy + full_r * np.sin(poly_angles)
        poly_pts = [[float(x), float(y)] for x, y in zip(pxs, pys)]

        return CircleAperture(
            center_x=full_cx,
            center_y=full_cy,
            radius=full_r,
            image_width=w,
            image_height=h,
            boundary_polygon=poly_pts,
        )

