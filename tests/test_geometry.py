"""Unit tests for geometric validation and transform storage."""

from pathlib import Path
import numpy as np
import pytest
from portpan.core.models import LocationTransforms, RegistrationResult, RegistrationStatus, TransformType
from portpan.geometry.transform_store import TransformStore
from portpan.geometry.validator import GeometricValidator


def test_validator_valid_affine():
    validator = GeometricValidator(min_inliers=20, min_inlier_ratio=0.2)
    # Affine scaling + translation matrix
    H = np.array([[0.2, 0.0, 500.0], [0.0, 0.2, 300.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    is_valid, msg, corners = validator.validate_transform(
        H, src_dim=(1000, 1000), canvas_dim=(5000, 3000), inlier_count=50, inlier_ratio=0.6
    )
    assert is_valid is True
    assert corners is not None
    assert len(corners) == 4


def test_validator_low_inliers():
    validator = GeometricValidator(min_inliers=30)
    H = np.eye(3)
    is_valid, msg, corners = validator.validate_transform(
        H, src_dim=(1000, 1000), canvas_dim=(5000, 3000), inlier_count=10, inlier_ratio=0.5
    )
    assert is_valid is False
    assert "Insufficient inliers" in msg


def test_validator_concave_bowtie():
    validator = GeometricValidator(min_inliers=10, min_inlier_ratio=0.1)
    # Degenerate matrix that flips coordinates crossing corners
    # Let corners be mapped to a bowtie
    # We can test by providing an invalid perspective matrix
    H_bad = np.array([[-1.0, 0.0, 500.0], [0.0, 1.0, 300.0], [0.0, 0.0, 1.0]])  # Negative det
    is_valid, msg, _ = validator.validate_transform(
        H_bad, src_dim=(1000, 1000), canvas_dim=(5000, 3000), inlier_count=50, inlier_ratio=0.5
    )
    assert is_valid is False


def test_transform_store(tmp_path: Path):
    loc_dir = tmp_path / "test_loc"
    loc_dir.mkdir()

    res = RegistrationResult(
        target_image_name="shot1.jpg",
        status=RegistrationStatus.SUCCESS,
        transform_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        transform_type=TransformType.HOMOGRAPHY,
        inlier_count=45,
        total_matches=80,
        inlier_ratio=0.56,
        confidence_score=0.9,
    )

    TransformStore.update_result(loc_dir, "test_loc", "base.jpg", 4000, 3000, res)

    loaded = TransformStore.load(loc_dir)
    assert loaded is not None
    assert loaded.location_name == "test_loc"
    assert "shot1.jpg" in loaded.transforms
    assert loaded.transforms["shot1.jpg"].inlier_count == 45
