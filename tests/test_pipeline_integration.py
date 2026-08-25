"""Integration test for the registration pipeline on real dataset samples."""

from pathlib import Path
import pytest
from portpan.core.models import RegistrationStatus
from portpan.geometry.transform_store import TransformStore
from portpan.pipeline.orchestrator import RegistrationPipeline


def test_pipeline_kolmer_gulch_integration(tmp_path: Path):
    loc_dir = Path("data/locations/kolmer_gulch")
    if not loc_dir.exists():
        pytest.skip("data/locations/kolmer_gulch not found")

    audit_dir = tmp_path / "audit"
    pipeline = RegistrationPipeline(audit_dir=audit_dir)

    # Process location
    transforms = pipeline.process_location(loc_dir, save_transforms=True, generate_audit_images=True)

    assert transforms.location_name == "kolmer_gulch"
    assert len(transforms.transforms) > 0
    assert (loc_dir / "transforms.json").exists()

    # Verify at least one image successfully registered
    successes = [r for r in transforms.transforms.values() if r.status in [RegistrationStatus.SUCCESS, RegistrationStatus.FALLBACK_AFFINE]]
    print(f"Total processed: {len(transforms.transforms)}, Successes: {len(successes)}")
    assert len(successes) > 0
