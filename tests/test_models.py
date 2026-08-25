"""Unit tests for Pydantic models in portpan.core.models."""

import numpy as np
import pytest
from portpan.core.models import (
    BoundingBox,
    CircleAperture,
    LocationTransforms,
    RegistrationResult,
    RegistrationStatus,
    TransformType,
)


def test_bounding_box_valid():
    box = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.8, xmax=0.9, confidence=0.95)
    assert box.width == pytest.approx(0.7)
    assert box.height == pytest.approx(0.7)
    assert box.area == pytest.approx(0.49)

    x1, y1, x2, y2 = box.to_pixel_coords(1000, 500)
    assert x1 == 200
    assert y1 == 50
    assert x2 == 900
    assert y2 == 400


def test_bounding_box_pad():
    box = BoundingBox(ymin=0.2, xmin=0.2, ymax=0.4, xmax=0.4)
    padded = box.pad(0.5)
    assert padded.xmin == pytest.approx(0.1)
    assert padded.ymin == pytest.approx(0.1)
    assert padded.xmax == pytest.approx(0.5)
    assert padded.ymax == pytest.approx(0.5)


def test_bounding_box_invalid_order():
    with pytest.raises(ValueError):
        BoundingBox(ymin=0.8, xmin=0.2, ymax=0.1, xmax=0.9)


def test_circle_aperture_inscribed_square():
    aperture = CircleAperture(center_x=500.0, center_y=500.0, radius=200.0, image_width=1000, image_height=1000)
    x1, y1, x2, y2 = aperture.get_inscribed_square(safety_margin=0.0)
    # radius = 200, half_side = 200 / sqrt(2) ≈ 141.42
    assert x1 == 358
    assert y1 == 358
    assert x2 == 642
    assert y2 == 642

    mask = aperture.generate_mask(feather_px=0)
    assert mask.shape == (1000, 1000)
    assert mask[500, 500] == 255
    assert mask[0, 0] == 0


def test_registration_result_serialization():
    res = RegistrationResult(
        target_image_name="test.jpg",
        status=RegistrationStatus.SUCCESS,
        transform_matrix=[[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
        transform_type=TransformType.HOMOGRAPHY,
        inlier_count=50,
        total_matches=100,
        inlier_ratio=0.5,
        confidence_score=0.85,
    )
    M = res.get_matrix_np()
    assert M.shape == (3, 3)
    assert M[0, 2] == 10.0
