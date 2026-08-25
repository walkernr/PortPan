"""Geometric validation and transform sanity checks."""

from __future__ import annotations

import cv2
import numpy as np


class GeometricValidator:
    """Validates estimated homography and affine matrices to prevent distorted or degenerate mappings."""

    def __init__(
        self,
        min_inliers: int = 25,
        min_inlier_ratio: float = 0.18,
        min_area_ratio: float = 0.00005,
        max_area_ratio: float = 0.35,
        max_aspect_deviation: float = 3.5,
        max_condition_number: float = 50.0,
    ) -> None:
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_aspect_deviation = max_aspect_deviation
        self.max_condition_number = max_condition_number

    def validate_transform(
        self,
        H: np.ndarray,
        src_dim: tuple[int, int],  # (width, height)
        canvas_dim: tuple[int, int],  # (width, height)
        inlier_count: int,
        inlier_ratio: float,
    ) -> tuple[bool, str, list[list[float]] | None]:
        """Validate an estimated 3x3 homography matrix.

        Returns:
            Tuple of (is_valid, reason_message, list_of_4_warped_corners).
        """
        # 1. Check inliers
        if inlier_count < self.min_inliers:
            return False, f"Insufficient inliers ({inlier_count} < {self.min_inliers})", None

        # Adaptive inlier ratio check: large absolute inlier counts (e.g. >= 25) have near-zero false positive probability
        effective_min_ratio = self.min_inlier_ratio
        if inlier_count >= 50:
            effective_min_ratio = min(effective_min_ratio, 0.06)
        elif inlier_count >= 25:
            effective_min_ratio = min(effective_min_ratio, 0.075)
        elif inlier_count >= 15:
            effective_min_ratio = min(effective_min_ratio, 0.12)

        if inlier_ratio < effective_min_ratio:
            return False, f"Inlier ratio too low ({inlier_ratio:.3f} < {effective_min_ratio:.3f})", None

        # 2. Check matrix finiteness
        if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
            return False, "Transform matrix is invalid or non-finite", None

        # Normalize matrix so H[2, 2] == 1.0
        if abs(H[2, 2]) < 1e-9:
            return False, "Matrix H[2,2] is near zero", None
        H_norm = H / H[2, 2]

        # 3. Check determinant
        det = np.linalg.det(H_norm)
        if det <= 1e-7:
            return False, f"Determinant is non-positive or degenerate ({det:.2e})", None

        # 4. Check condition number of 2x2 affine portion
        u, s, vh = np.linalg.svd(H_norm[:2, :2])
        if s[1] < 1e-7:
            return False, "Transform has near-zero singular value", None
        cond_num = s[0] / s[1]
        if cond_num > self.max_condition_number:
            return False, f"Condition number too high ({cond_num:.1f} > {self.max_condition_number})", None

        # 5. Warp the 4 corners of the source image
        src_w, src_h = src_dim
        canvas_w, canvas_h = canvas_dim
        src_corners = np.array(
            [[[0.0, 0.0]], [[src_w, 0.0]], [[src_w, src_h]], [[0.0, src_h]]], dtype=np.float32
        )

        try:
            dst_corners = cv2.perspectiveTransform(src_corners, H_norm).reshape(-1, 2)
        except Exception as e:
            return False, f"Perspective transform failed: {e}", None

        corners_list = [[float(pt[0]), float(pt[1])] for pt in dst_corners]

        # 6. Check Convexity (no self-intersecting bowties)
        # Compute cross product of consecutive edge vectors
        edges = []
        for i in range(4):
            p1 = dst_corners[i]
            p2 = dst_corners[(i + 1) % 4]
            edges.append(p2 - p1)

        cross_signs = []
        for i in range(4):
            e1 = edges[i]
            e2 = edges[(i + 1) % 4]
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            cross_signs.append(cross > 0)

        # In a strictly convex polygon, all cross products must have identical sign
        if not (all(cross_signs) or not any(cross_signs)):
            return False, "Warped quadrilateral is concave or self-intersecting (bowtie)", corners_list

        # 7. Check Area Sanity
        # Shoelace formula for polygon area
        x = dst_corners[:, 0]
        y = dst_corners[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        canvas_area = float(canvas_w * canvas_h)
        area_ratio = area / canvas_area

        if area_ratio < self.min_area_ratio:
            return False, f"Warped area too small ({area_ratio:.2e} < {self.min_area_ratio:.2e})", corners_list

        if area_ratio > self.max_area_ratio:
            return False, f"Warped area too large ({area_ratio:.2f} > {self.max_area_ratio:.2f})", corners_list

        # 8. Check Aspect Ratio Skew
        # Edge lengths: top, right, bottom, left
        d_top = np.linalg.norm(edges[0])
        d_right = np.linalg.norm(edges[1])
        d_bottom = np.linalg.norm(edges[2])
        d_left = np.linalg.norm(edges[3])

        avg_w = (d_top + d_bottom) / 2.0
        avg_h = (d_left + d_right) / 2.0

        if avg_h < 1e-4:
            return False, "Warped height is near zero", corners_list

        warped_aspect = avg_w / avg_h
        orig_aspect = src_w / src_h
        aspect_ratio_deviation = max(warped_aspect / orig_aspect, orig_aspect / warped_aspect)

        if aspect_ratio_deviation > self.max_aspect_deviation:
            return (
                False,
                f"Aspect ratio skew too large ({aspect_ratio_deviation:.2f} > {self.max_aspect_deviation})",
                corners_list,
            )

        return True, "Valid homography", corners_list
