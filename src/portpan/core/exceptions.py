"""Domain exception hierarchy for PortPan."""

from typing import Any


class PortPanError(Exception):
    """Base exception for all PortPan pipeline errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ApertureDetectionError(PortPanError):
    """Raised when optical circle aperture segmentation fails or produces invalid geometry."""


class LocalizationError(PortPanError):
    """Raised when semantic/coarse localization cannot find the target region."""


class MatchingError(PortPanError):
    """Raised when dense feature matching fails or produces insufficient inliers."""


class GeometricSanityError(PortPanError):
    """Raised when an estimated homography/affine transformation violates sanity constraints."""


class TransformStorageError(PortPanError):
    """Raised when reading or writing transforms.json fails."""


class VideoProcessingError(PortPanError):
    """Raised during Live Photo video ingestion, loop generation, or frame synthesis."""
