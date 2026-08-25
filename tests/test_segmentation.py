"""Unit tests for aperture segmentation."""

import cv2
import numpy as np
import pytest
from portpan.segmentation.circle_segmenter import RobustCircleSegmenter


def test_segmenter_synthetic_circle():
    # Create black canvas 1000x800 with white circle
    img = np.zeros((800, 1000, 3), dtype=np.uint8)
    cv2.circle(img, (520, 390), 300, (200, 200, 200), -1)

    segmenter = RobustCircleSegmenter()
    aperture = segmenter.segment(img)

    assert aperture.center_x == pytest.approx(520, abs=15)
    assert aperture.center_y == pytest.approx(390, abs=15)
    assert aperture.radius == pytest.approx(300, abs=20)

    crop, bounds = segmenter.extract_matching_target(img, aperture)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_segmenter_full_frame():
    # Uniform white image with bright corners
    img = np.full((600, 800, 3), 180, dtype=np.uint8)
    segmenter = RobustCircleSegmenter()
    aperture = segmenter.segment(img)

    assert aperture.center_x == pytest.approx(400, abs=5)
    assert aperture.center_y == pytest.approx(300, abs=5)
