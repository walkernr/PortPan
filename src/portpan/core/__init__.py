"""Core domain models, exceptions, and auditing utilities."""

from portpan.core.audit import VisualAuditor
from portpan.core.exceptions import (
    ApertureDetectionError,
    GeometricSanityError,
    LocalizationError,
    MatchingError,
    PortPanError,
    TransformStorageError,
    VideoProcessingError,
)
from portpan.core.models import (
    BoundingBox,
    CircleAperture,
    LocationTransforms,
    RegistrationResult,
    RegistrationStatus,
    TransformType,
)

__all__ = [
    "ApertureDetectionError",
    "BoundingBox",
    "CircleAperture",
    "GeometricSanityError",
    "LocationTransforms",
    "LocalizationError",
    "MatchingError",
    "PortPanError",
    "RegistrationResult",
    "RegistrationStatus",
    "TransformStorageError",
    "TransformType",
    "VideoProcessingError",
    "VisualAuditor",
]
