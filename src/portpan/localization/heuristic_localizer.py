"""Offline heuristic and multi-tile feature localizer for local processing and testing."""

from __future__ import annotations

import cv2
import numpy as np
from portpan.core.models import BoundingBox
from portpan.localization.base import BaseCoarseLocalizer


class HeuristicLocalizer(BaseCoarseLocalizer):
    """Offline localizer using dense SIFT feature clustering and sliding window search."""

    def __init__(self, n_features: int = 15000) -> None:
        self.n_features = n_features

    def localize(self, base_image: np.ndarray, target_patch: np.ndarray) -> list[BoundingBox]:
        """Locate candidate bounding boxes in base_image using fast feature matching."""
        h_base, w_base = base_image.shape[:2]
        gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY) if len(base_image.shape) == 3 else base_image
        gray_tgt = cv2.cvtColor(target_patch, cv2.COLOR_BGR2GRAY) if len(target_patch.shape) == 3 else target_patch

        sift = cv2.SIFT_create(nfeatures=self.n_features, contrastThreshold=0.025)
        kp_tgt, des_tgt = sift.detectAndCompute(gray_tgt, None)
        kp_base, des_base = sift.detectAndCompute(gray_base, None)

        if des_tgt is None or des_base is None or len(kp_tgt) < 8 or len(kp_base) < 8:
            return [BoundingBox(ymin=0.0, xmin=0.0, ymax=1.0, xmax=1.0, confidence=0.5)]

        # FLANN matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(des_tgt, des_base, k=2)
        except Exception:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des_tgt, des_base, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.78 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 6:
            # Fallback to full frame
            return [BoundingBox(ymin=0.0, xmin=0.0, ymax=1.0, xmax=1.0, confidence=0.3)]

        pts_base = np.float32([kp_base[m.trainIdx].pt for m in good_matches])

        # Cluster points in base space (DBSCAN / percentile bounding)
        # Use robust percentile to eliminate rogue outlier matches
        x_coords = pts_base[:, 0]
        y_coords = pts_base[:, 1]

        # Use 10th and 90th percentiles to define core cluster
        p_x1, p_x2 = float(np.percentile(x_coords, 5)), float(np.percentile(x_coords, 95))
        p_y1, p_y2 = float(np.percentile(y_coords, 5)), float(np.percentile(y_coords, 95))

        # Add margin
        span_x = max(100.0, p_x2 - p_x1)
        span_y = max(100.0, p_y2 - p_y1)

        x1 = max(0, int(p_x1 - span_x * 0.25))
        y1 = max(0, int(p_y1 - span_y * 0.25))
        x2 = min(w_base, int(p_x2 + span_x * 0.25))
        y2 = min(h_base, int(p_y2 + span_y * 0.25))

        primary_box = BoundingBox.from_pixel_coords(
            x1, y1, x2, y2, w_base, h_base, confidence=min(1.0, len(good_matches) / 30.0)
        )

        # Also provide full frame as backup candidate
        full_box = BoundingBox(ymin=0.0, xmin=0.0, ymax=1.0, xmax=1.0, confidence=0.3)

        return [primary_box, full_box]
