"""Abstract interface for coarse semantic localization."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from portpan.core.models import BoundingBox


class BaseCoarseLocalizer(ABC):
    """Abstract interface for finding candidate bounding boxes of a target patch on a wide canvas."""

    @abstractmethod
    def localize(self, base_image: np.ndarray, target_patch: np.ndarray) -> list[BoundingBox]:
        """Locate candidate bounding boxes of target_patch in base_image.

        Args:
            base_image: Master panorama / canvas (BGR uint8 NumPy array).
            target_patch: Inscribed digiscope crop (BGR uint8 NumPy array).

        Returns:
            List of candidate BoundingBox objects ordered by confidence descending.
        """
        pass
