"""Abstract interface for dense sub-pixel feature matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np


class MatchResult(NamedTuple):
    """Result of feature matching between a candidate base crop and target digiscope patch."""

    H_local: np.ndarray | None
    pts_base: np.ndarray
    pts_target: np.ndarray
    inlier_mask: np.ndarray | None
    inlier_count: int
    inlier_ratio: float
    is_affine_fallback: bool = False


class BaseDenseMatcher(ABC):
    """Abstract interface for dense feature matching and homography/affine estimation."""

    @abstractmethod
    def match(self, base_crop: np.ndarray, target_patch: np.ndarray) -> MatchResult:
        """Match keypoints between base_crop and target_patch and estimate local 3x3 transform.

        Args:
            base_crop: High-resolution candidate sub-patch from master canvas.
            target_patch: Inscribed digiscope observation patch.

        Returns:
            MatchResult containing local transform, keypoints, inlier mask, count, and ratio.
        """
        pass
