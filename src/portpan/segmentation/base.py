"""Abstract base class for optical aperture segmentation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import cv2
import numpy as np
from portpan.core.models import CircleAperture


class BaseApertureSegmenter(ABC):
    """Abstract interface for detecting optical circular apertures in digiscope captures."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> CircleAperture:
        """Detect the optical circular aperture within the given image.

        Args:
            image: BGR uint8 NumPy array.

        Returns:
            CircleAperture model with center coordinates, radius, and image dimensions.

        Raises:
            ApertureDetectionError: If segmentation fails or aperture is degenerate.
        """
        pass

    def extract_matching_target(
        self, image: np.ndarray, aperture: CircleAperture, safety_margin: float = 0.05
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Extract the inscribed square crop free from barrel vignette and outer optical blur.

        Returns:
            Tuple of (cropped_image, (x1, y1, x2, y2)).
        """
        x1, y1, x2, y2 = aperture.get_inscribed_square(safety_margin=safety_margin)
        crop = image[y1:y2, x1:x2].copy()
        return crop, (x1, y1, x2, y2)

    def extract_porthole_rgba(
        self, image: np.ndarray, aperture: CircleAperture, feather_px: int = 3
    ) -> np.ndarray:
        """Generate a 4-channel BGRA image with feathered transparent background outside the aperture."""
        h, w = image.shape[:2]
        mask = aperture.generate_mask(feather_px=feather_px)

        if image.shape[2] == 3:
            b, g, r = cv2.split(image)
            rgba = cv2.merge([b, g, r, mask])
        else:
            rgba = image.copy()
            rgba[:, :, 3] = mask

        return rgba
