"""Pydantic data models and schemas for PortPan."""

from __future__ import annotations

from enum import Enum
import math
from typing import Any
import numpy as np
from pydantic import BaseModel, Field, field_validator


class TransformType(str, Enum):
    """Type of 2D geometric transformation."""

    HOMOGRAPHY = "HOMOGRAPHY"
    AFFINE = "AFFINE"
    SIMILARITY = "SIMILARITY"
    TRANSLATION = "TRANSLATION"


class RegistrationStatus(str, Enum):
    """Status of image-to-canvas registration."""

    SUCCESS = "SUCCESS"
    FALLBACK_AFFINE = "FALLBACK_AFFINE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FAILED = "FAILED"


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates in range [0.0, 1.0].

    Coordinates follow the standard computer vision format:
    ymin, xmin, ymax, xmax.
    """

    ymin: float = Field(ge=0.0, le=1.0, description="Normalized top coordinate")
    xmin: float = Field(ge=0.0, le=1.0, description="Normalized left coordinate")
    ymax: float = Field(ge=0.0, le=1.0, description="Normalized bottom coordinate")
    xmax: float = Field(ge=0.0, le=1.0, description="Normalized right coordinate")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Localization confidence score")

    @field_validator("ymax")
    @classmethod
    def validate_y_order(cls, v: float, info: Any) -> float:
        """Ensure ymax >= ymin."""
        ymin = info.data.get("ymin", 0.0)
        if v < ymin:
            raise ValueError(f"ymax ({v}) must be greater than or equal to ymin ({ymin})")
        return v

    @field_validator("xmax")
    @classmethod
    def validate_x_order(cls, v: float, info: Any) -> float:
        """Ensure xmax >= xmin."""
        xmin = info.data.get("xmin", 0.0)
        if v < xmin:
            raise ValueError(f"xmax ({v}) must be greater than or equal to xmin ({xmin})")
        return v

    @property
    def width(self) -> float:
        """Normalized width."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Normalized height."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Normalized area."""
        return self.width * self.height

    def to_pixel_coords(self, canvas_width: int, canvas_height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute integer pixel bounds [x1, y1, x2, y2]."""
        x1 = max(0, min(canvas_width - 1, int(round(self.xmin * canvas_width))))
        y1 = max(0, min(canvas_height - 1, int(round(self.ymin * canvas_height))))
        x2 = max(x1 + 1, min(canvas_width, int(round(self.xmax * canvas_width))))
        y2 = max(y1 + 1, min(canvas_height, int(round(self.ymax * canvas_height))))
        return (x1, y1, x2, y2)

    def pad(self, factor: float = 0.25) -> BoundingBox:
        """Return a new bounding box expanded symmetrically by the given factor (e.g. 0.25 = +25%)."""
        pad_w = self.width * factor
        pad_h = self.height * factor
        return BoundingBox(
            ymin=max(0.0, self.ymin - pad_h),
            xmin=max(0.0, self.xmin - pad_w),
            ymax=min(1.0, self.ymax + pad_h),
            xmax=min(1.0, self.xmax + pad_w),
            confidence=self.confidence,
        )

    @classmethod
    def from_pixel_coords(
        cls, x1: int, y1: int, x2: int, y2: int, canvas_width: int, canvas_height: int, confidence: float = 1.0
    ) -> BoundingBox:
        """Construct a normalized BoundingBox from integer pixel coordinates."""
        return cls(
            ymin=max(0.0, min(1.0, y1 / canvas_height)),
            xmin=max(0.0, min(1.0, x1 / canvas_width)),
            ymax=max(0.0, min(1.0, y2 / canvas_height)),
            xmax=max(0.0, min(1.0, x2 / canvas_width)),
            confidence=confidence,
        )


class CircleAperture(BaseModel):
    """Detected circular optical aperture in a digiscope capture."""

    center_x: float = Field(ge=0.0, description="Center X coordinate in pixels")
    center_y: float = Field(ge=0.0, description="Center Y coordinate in pixels")
    radius: float = Field(gt=0.0, description="Radius in pixels")
    image_width: int = Field(gt=0, description="Width of source image")
    image_height: int = Field(gt=0, description="Height of source image")

    def get_inscribed_square(self, safety_margin: float = 0.05) -> tuple[int, int, int, int]:
        """Compute the pixel coordinates (x1, y1, x2, y2) of the largest square inside the circle.

        Args:
            safety_margin: Fraction of radius to shrink inwardly to stay clear of vignette falloff.

        Returns:
            Tuple of integer pixel bounds (x1, y1, x2, y2).
        """
        effective_radius = self.radius * (1.0 - safety_margin)
        # Side length of inscribed square = sqrt(2) * r, half side = r / sqrt(2)
        half_side = effective_radius / math.sqrt(2.0)

        x1 = max(0, int(math.floor(self.center_x - half_side)))
        y1 = max(0, int(math.floor(self.center_y - half_side)))
        x2 = min(self.image_width, int(math.ceil(self.center_x + half_side)))
        y2 = min(self.image_height, int(math.ceil(self.center_y + half_side)))

        return (x1, y1, x2, y2)

    def generate_mask(self, feather_px: int = 3) -> np.ndarray:
        """Generate a 2D uint8 alpha mask (0-255) with anti-aliased / feathered circular edge."""
        import cv2

        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        center = (int(round(self.center_x)), int(round(self.center_y)))
        radius_int = int(round(self.radius))

        # Draw solid circle
        cv2.circle(mask, center, radius_int, 255, -1, lineType=cv2.LINE_AA)

        if feather_px > 0:
            # Gaussian blur for soft anti-aliased alpha edge
            ksize = feather_px * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_px / 2.0)

        return mask


class RegistrationResult(BaseModel):
    """Result of mapping a single digiscope image onto the base panorama."""

    target_image_name: str = Field(description="Filename of the digiscope image")
    status: RegistrationStatus = Field(description="Status of the registration")
    transform_matrix: list[list[float]] = Field(description="3x3 global homography / affine matrix")
    transform_type: TransformType = Field(default=TransformType.HOMOGRAPHY, description="Transform type used")
    inlier_count: int = Field(ge=0, description="Number of RANSAC/MAGSAC inliers")
    total_matches: int = Field(ge=0, description="Total candidate feature matches")
    inlier_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of inliers to matches")
    reprojection_error: float | None = Field(default=None, description="Mean RMSE reprojection error")
    bounding_box_in_base: list[list[float]] | None = Field(
        default=None, description="4 warped corner coordinates in base canvas space [[x, y], ...]"
    )
    confidence_score: float = Field(ge=0.0, le=1.0, description="Estimated registration confidence")
    video_to_still_transform: list[list[float]] | None = Field(
        default=None, description="3x3 matrix mapping video keyframe to still JPEG (for EIS correction)"
    )
    notes: str | None = Field(default=None, description="Diagnostic notes or fallback reasons")

    def get_matrix_np(self) -> np.ndarray:
        """Return the 3x3 transform as a float64 NumPy array."""
        return np.array(self.transform_matrix, dtype=np.float64)

    def get_video_matrix_np(self) -> np.ndarray | None:
        """Return the 3x3 video-to-still matrix as a float64 NumPy array if present."""
        if self.video_to_still_transform is None:
            return None
        return np.array(self.video_to_still_transform, dtype=np.float64)


class LocationTransforms(BaseModel):
    """Container for all transform registrations in a single location dataset."""

    location_name: str = Field(description="Directory/location identifier")
    base_image_name: str = Field(description="Filename of the master base canvas image")
    base_width: int = Field(gt=0, description="Master canvas pixel width")
    base_height: int = Field(gt=0, description="Master canvas pixel height")
    created_at: str = Field(description="ISO 8601 timestamp of creation/update")
    transforms: dict[str, RegistrationResult] = Field(
        default_factory=dict, description="Map of digiscope image filename to RegistrationResult"
    )
