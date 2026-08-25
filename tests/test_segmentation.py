"""Unit tests for aperture segmentation."""

import cv2
import numpy as np
import pytest
from portpan.segmentation.circle_segmenter import RobustCircleSegmenter
from portpan.segmentation.polar_segmenter import PolarChromaSegmenter


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


def test_polar_chroma_segmenter_synthetic():
    # Create canvas with chromatic aberration ring
    img = np.zeros((800, 1000, 3), dtype=np.uint8)
    # Core scene (neutral)
    cv2.circle(img, (500, 400), 280, (180, 180, 180), -1)
    # Outer chromatic fringe (blue/cyan ring)
    cv2.circle(img, (500, 400), 310, (240, 150, 40), 25)

    segmenter = PolarChromaSegmenter()
    aperture = segmenter.segment(img)

    assert aperture.center_x == pytest.approx(500, abs=15)
    assert aperture.center_y == pytest.approx(400, abs=15)
    assert aperture.boundary_polygon is not None
    assert len(aperture.boundary_polygon) == 360

    # Test RGBA extraction
    rgba = aperture.extract_rgba(img, feather_px=2.0)
    assert rgba.shape == (800, 1000, 4)
    # Center should be opaque
    assert rgba[400, 500, 3] == 255
    # Outer corners should be completely transparent
    assert rgba[10, 10, 3] == 0
