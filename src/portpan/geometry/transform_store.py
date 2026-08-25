"""JSON serialization and storage management for location transforms."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from portpan.core.exceptions import TransformStorageError
from portpan.core.models import LocationTransforms, RegistrationResult


class TransformStore:
    """Manages reading and writing data/locations/<location_name>/transforms.json."""

    @staticmethod
    def get_transforms_path(location_dir: Path | str) -> Path:
        """Return the path to transforms.json in the location directory."""
        return Path(location_dir) / "transforms.json"

    @classmethod
    def load(cls, location_dir: Path | str) -> LocationTransforms | None:
        """Load LocationTransforms from transforms.json if it exists."""
        path = cls.get_transforms_path(location_dir)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return LocationTransforms.model_validate(data)
        except Exception as e:
            raise TransformStorageError(f"Failed to load transforms from {path}: {e}") from e

    @classmethod
    def save(cls, location_dir: Path | str, transforms: LocationTransforms) -> Path:
        """Save LocationTransforms to transforms.json atomically."""
        path = cls.get_transforms_path(location_dir)
        temp_path = path.with_suffix(".tmp.json")

        try:
            payload = transforms.model_dump(mode="json")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            temp_path.replace(path)
            return path
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise TransformStorageError(f"Failed to save transforms to {path}: {e}") from e

    @classmethod
    def update_result(
        cls,
        location_dir: Path | str,
        location_name: str,
        base_image_name: str,
        base_width: int,
        base_height: int,
        result: RegistrationResult,
    ) -> LocationTransforms:
        """Add or update a single RegistrationResult in transforms.json."""
        existing = cls.load(location_dir)
        now_str = datetime.now(timezone.utc).isoformat()

        if existing is None:
            existing = LocationTransforms(
                location_name=location_name,
                base_image_name=base_image_name,
                base_width=base_width,
                base_height=base_height,
                created_at=now_str,
                transforms={},
            )

        existing.transforms[result.target_image_name] = result
        cls.save(location_dir, existing)
        return existing
