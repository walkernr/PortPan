"""Unit tests for dense feature matching."""

import cv2
import numpy as np
import pytest
from portpan.matching.opencv_matcher import OpenCVMatcher


def test_opencv_matcher_synthetic_warp():
    # Create textured image with random noise and distinct shapes
    np.random.seed(42)
    base = np.random.randint(50, 200, (600, 800, 3), dtype=np.uint8)
    for _ in range(30):
        cx, cy = np.random.randint(50, 750), np.random.randint(50, 550)
        r = np.random.randint(10, 40)
        color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
        cv2.circle(base, (cx, cy), r, color, -1)

    # Crop a sub-region (target)
    crop = base[100:400, 150:450].copy()

    matcher = OpenCVMatcher(algorithm="SIFT", n_features=2000)
    result = matcher.match(base, crop)

    assert result.H_local is not None
    assert result.inlier_count >= 15
    assert result.inlier_ratio > 0.3
