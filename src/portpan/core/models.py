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
    """Detected optical aperture in a digiscope capture with sub-pixel chroma boundary polygon."""

    center_x: float = Field(ge=0.0, description="Center X coordinate in pixels")
    center_y: float = Field(ge=0.0, description="Center Y coordinate in pixels")
    radius: float = Field(gt=0.0, description="Effective mean radius in pixels")
    image_width: int = Field(gt=0, description="Width of source image")
    image_height: int = Field(gt=0, description="Height of source image")
    boundary_polygon: list[list[float]] | None = Field(
        default=None, description="Sub-pixel boundary contour coordinates [[x0, y0], [x1, y1], ...] around chromatic ring"
    )
    ellipse_axes: tuple[float, float] | None = Field(
        default=None, description="Fitted ellipse (major_axis, minor_axis) in pixels"
    )
    ellipse_angle: float | None = Field(
        default=None, description="Fitted ellipse rotation angle in degrees"
    )

    def get_inscribed_square(self, safety_margin: float = 0.05) -> tuple[int, int, int, int]:
        """Compute the pixel coordinates (x1, y1, x2, y2) of the largest square inside the aperture.

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

    @staticmethod
    def apply_luma_modulation(
        mask: np.ndarray, image: np.ndarray, rim_fraction: float = 0.15
    ) -> np.ndarray:
        """Apply adaptive chroma- and luma-aware transparency modulation to an alpha mask.

        Fades dark bevel regions to transparent in the outer rim zone so they
        don't occlude overlapping porthole detail, while keeping the vibrant
        blue/cyan chromatic fringe fully opaque. Works on both uint8 (0-255)
        and float32 (0.0-1.0) masks.

        Args:
            mask: 2D alpha mask (uint8 0-255 or float32 0.0-1.0).
            image: BGR source image for luma and chromaticity extraction.
            rim_fraction: Fraction of mask radius defining the outer rim zone.

        Returns:
            Modulated mask in the same dtype as input.
        """
        import cv2

        is_float = mask.dtype == np.float32
        if is_float:
            mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            mask_u8 = mask

        if len(image.shape) == 3 and image.shape[2] == 3:
            b = image[:, :, 0].astype(np.float32)
            g = image[:, :, 1].astype(np.float32)
            r = image[:, :, 2].astype(np.float32)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Blue chromatic excess indicates the true optical boundary fringe
            blue_excess = b - 0.5 * (r + g)
        else:
            gray = image
            blue_excess = np.zeros_like(gray, dtype=np.float32)

        # Distance transform to apply modulation in the outer rim zone
        dist = cv2.distanceTransform((mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        max_d = float(np.max(dist)) if np.max(dist) > 0 else 1.0

        # Interior reference luminance (median of central aperture core)
        interior_mask = dist > (max_d * 0.35)
        if np.any(interior_mask):
            luma_ref = float(np.median(gray[interior_mask]))
        else:
            luma_ref = 120.0

        # Rim factor: 0.0 at inner rim boundary, 1.0 at outer mask perimeter
        rim_factor = np.clip(1.0 - (dist / (max_d * rim_fraction + 1e-5)), 0.0, 1.0)

        # Chromatic boost preserves the cyan/blue dispersion ring
        chroma_boost = np.clip((blue_excess - 5.0) / 25.0, 0.0, 1.0)

        # Dark bevel falloff: dark eyecup plastic (gray 20-75) fades to 0
        luma_thresh_high = max(55.0, luma_ref * 0.50)
        luma_thresh_low = 22.0
        luma_alpha = np.clip((gray.astype(np.float32) - luma_thresh_low) / (luma_thresh_high - luma_thresh_low + 1e-5), 0.0, 1.0)

        # Effective alpha in rim: opaque if bright OR if chromatic blue ring
        effective_alpha = np.maximum(chroma_boost, luma_alpha)

        mask_f = (mask_u8.astype(np.float32) / 255.0) * (1.0 - rim_factor * (1.0 - effective_alpha))

        if is_float:
            return np.clip(mask_f, 0.0, 1.0).astype(np.float32)
        return (np.clip(mask_f, 0.0, 1.0) * 255.0).astype(np.uint8)

    def generate_mask(self, feather_px: float = 2.0, image: np.ndarray | None = None) -> np.ndarray:
        """Generate a 2D uint8 alpha mask (0-255) preserving the chromatic aberration ring with anti-aliasing."""
        import cv2

        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        if self.boundary_polygon and len(self.boundary_polygon) >= 3:
            pts = np.array(self.boundary_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA)
        else:
            center = (int(round(self.center_x)), int(round(self.center_y)))
            radius_int = int(round(self.radius))
            cv2.circle(mask, center, radius_int, 255, -1, lineType=cv2.LINE_AA)

        if feather_px > 0:
            ksize = int(round(feather_px * 2)) | 1  # ensure odd kernel size
            mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_px / 2.0)

        if image is not None:
            mask = CircleAperture.apply_luma_modulation(mask, image)

        return mask

    def extract_rgba(self, image: np.ndarray, feather_px: float = 2.0) -> np.ndarray:
        """Extract a transparent 4-channel BGRA/RGBA porthole image retaining the full chromatic boundary."""
        import cv2

        mask = self.generate_mask(feather_px=feather_px, image=image)
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) if len(image.shape) == 3 and image.shape[2] == 3 else image.copy()
        bgra[:, :, 3] = mask
        return bgra


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
