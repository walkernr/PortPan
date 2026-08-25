"""End-to-end registration pipeline orchestrator for PortPan."""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from portpan.core.audit import VisualAuditor
from portpan.core.exceptions import PortPanError
from portpan.core.models import (
    BoundingBox,
    CircleAperture,
    LocationTransforms,
    RegistrationResult,
    RegistrationStatus,
    TransformType,
)
from portpan.geometry.transform_store import TransformStore
from portpan.geometry.validator import GeometricValidator
from portpan.localization.base import BaseCoarseLocalizer
from portpan.localization.heuristic_localizer import HeuristicLocalizer
from portpan.matching.base import BaseDenseMatcher
from portpan.matching.opencv_matcher import OpenCVMatcher
from portpan.segmentation.base import BaseApertureSegmenter
from portpan.segmentation.circle_segmenter import RobustCircleSegmenter
from portpan.video.stabilizer_offset import VideoStabilizerOffsetEstimator

logger = logging.getLogger("portpan.pipeline")


class RegistrationPipeline:
    """Coordinates aperture segmentation, hierarchical coarse localization, dense feature matching, and validation."""

    def __init__(
        self,
        segmenter: BaseApertureSegmenter | None = None,
        localizer: BaseCoarseLocalizer | None = None,
        matcher: BaseDenseMatcher | None = None,
        validator: GeometricValidator | None = None,
        video_offset_estimator: VideoStabilizerOffsetEstimator | None = None,
        audit_dir: Path | str | None = "output/audit",
    ) -> None:
        self.segmenter = segmenter or RobustCircleSegmenter()
        self.localizer = localizer or HeuristicLocalizer()
        self.matcher = matcher or OpenCVMatcher()
        self.validator = validator or GeometricValidator()
        self.video_offset_estimator = video_offset_estimator or VideoStabilizerOffsetEstimator()
        self.auditor = VisualAuditor(audit_dir) if audit_dir else None

    def process_location(
        self,
        location_dir: Path | str,
        save_transforms: bool = True,
        generate_audit_images: bool = True,
    ) -> LocationTransforms:
        """Process all digiscope images and video streams in a location directory."""
        loc_path = Path(location_dir)
        loc_name = loc_path.name

        base_dir = loc_path / "base"
        images_dir = loc_path / "images"
        videos_dir = loc_path / "videos"

        # Find master base image
        base_files = list(base_dir.glob("*.jpg")) + list(base_dir.glob("*.png"))
        if not base_files:
            raise PortPanError(f"No base panorama image found in {base_dir}")
        base_path = sorted(base_files)[0]

        logger.info(f"Loading master canvas: {base_path.name}")
        base_canvas = cv2.imread(str(base_path))
        if base_canvas is None:
            raise PortPanError(f"Failed to load master canvas image: {base_path}")

        h_canvas, w_canvas = base_canvas.shape[:2]

        # Find digiscope images
        img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        if not img_files:
            logger.warning(f"No digiscope images found in {images_dir}")

        # Index available videos by base stem
        video_map: dict[str, Path] = {}
        if videos_dir.exists():
            for v in videos_dir.glob("*.mp4"):
                stem = v.name.split("_exported")[0]
                video_map[stem] = v

        results: dict[str, RegistrationResult] = {}
        images_map: dict[str, np.ndarray] = {}
        apertures_map: dict[str, CircleAperture] = {}

        for img_path in img_files:
            img_name = img_path.name
            logger.info(f"Registering digiscope asset: {img_name}")
            target_img = cv2.imread(str(img_path))
            if target_img is None:
                logger.error(f"Failed to read image: {img_path}")
                continue

            h_tgt, w_tgt = target_img.shape[:2]

            # Step 1: Aperture Segmentation & Inscribed Pre-Crop
            try:
                aperture = self.segmenter.segment(target_img)
                inscribed_crop, inscribed_box = self.segmenter.extract_matching_target(target_img, aperture)
            except Exception as e:
                logger.warning(f"Aperture segmentation failed on {img_name}: {e}")
                # Fallback to full frame
                aperture = CircleAperture(
                    center_x=w_tgt / 2.0,
                    center_y=h_tgt / 2.0,
                    radius=min(w_tgt, h_tgt) * 0.45,
                    image_width=w_tgt,
                    image_height=h_tgt,
                )
                inscribed_crop = target_img
                inscribed_box = (0, 0, w_tgt, h_tgt)

            if self.auditor and generate_audit_images:
                self.auditor.draw_aperture_debug(target_img, aperture, inscribed_box, f"{loc_name}_{img_path.stem}")

            # Step 2: Semantic Coarse Localization
            candidate_boxes = self.localizer.localize(base_canvas, inscribed_crop)

            best_reg_result: RegistrationResult | None = None

            # Step 3 & 4: Sub-pixel Dense Matching on Candidate Patches
            for box_idx, box in enumerate(candidate_boxes):
                padded_box = box.pad(factor=0.25)
                bx1, by1, bx2, by2 = padded_box.to_pixel_coords(w_canvas, h_canvas)
                base_crop = base_canvas[by1:by2, bx1:bx2]

                if base_crop.size == 0:
                    continue

                match_res = self.matcher.match(base_crop, inscribed_crop)

                if match_res.H_local is None or match_res.inlier_count < 10:
                    continue

                # Convert local homography H_local into global canvas coordinate space
                # Let p_tgt be in target_img coords, p_crop = p_tgt - (ix1, iy1)
                # Then p_base_crop = H_local @ p_crop
                # p_canvas = p_base_crop + (bx1, by1)
                ix1, iy1, ix2, iy2 = inscribed_box
                T_inscribed_inv = np.array(
                    [[1.0, 0.0, -float(ix1)], [0.0, 1.0, -float(iy1)], [0.0, 0.0, 1.0]], dtype=np.float64
                )
                T_base = np.array(
                    [[1.0, 0.0, float(bx1)], [0.0, 1.0, float(by1)], [0.0, 0.0, 1.0]], dtype=np.float64
                )

                H_global = T_base @ match_res.H_local @ T_inscribed_inv

                # Step 5: Geometric Validation
                is_valid, reason, warped_corners = self.validator.validate_transform(
                    H_global,
                    src_dim=(w_tgt, h_tgt),
                    canvas_dim=(w_canvas, h_canvas),
                    inlier_count=match_res.inlier_count,
                    inlier_ratio=match_res.inlier_ratio,
                )

                if is_valid:
                    # Optional: Estimate Video-to-Still stabilization offset
                    stem = img_path.name.split(".")[0]
                    H_vid_still: list[list[float]] | None = None
                    if stem in video_map:
                        try:
                            H_v2s = self.video_offset_estimator.estimate_offset(target_img, video_map[stem])
                            H_vid_still = H_v2s.tolist()
                        except Exception as e:
                            logger.warning(f"Video offset estimation failed for {stem}: {e}")

                    status = (
                        RegistrationStatus.FALLBACK_AFFINE
                        if match_res.is_affine_fallback
                        else RegistrationStatus.SUCCESS
                    )
                    transform_type = (
                        TransformType.AFFINE if match_res.is_affine_fallback else TransformType.HOMOGRAPHY
                    )

                    best_reg_result = RegistrationResult(
                        target_image_name=img_name,
                        status=status,
                        transform_matrix=H_global.tolist(),
                        transform_type=transform_type,
                        inlier_count=match_res.inlier_count,
                        total_matches=len(match_res.pts_target),
                        inlier_ratio=match_res.inlier_ratio,
                        bounding_box_in_base=warped_corners,
                        confidence_score=min(1.0, match_res.inlier_ratio * (match_res.inlier_count / 30.0)),
                        video_to_still_transform=H_vid_still,
                        notes=f"Matched on candidate tile {box_idx + 1}/{len(candidate_boxes)}",
                    )

                    if self.auditor and generate_audit_images and match_res.inlier_mask is not None:
                        self.auditor.draw_matches_debug(
                            base_crop,
                            inscribed_crop,
                            match_res.pts_base,
                            match_res.pts_target,
                            match_res.inlier_mask,
                            f"{loc_name}_{img_path.stem}",
                        )
                    break

            if best_reg_result is None:
                logger.warning(f"Registration failed for {img_name}")
                # Create a failed stub
                best_reg_result = RegistrationResult(
                    target_image_name=img_name,
                    status=RegistrationStatus.FAILED,
                    transform_matrix=np.eye(3).tolist(),
                    transform_type=TransformType.HOMOGRAPHY,
                    inlier_count=0,
                    total_matches=0,
                    inlier_ratio=0.0,
                    confidence_score=0.0,
                    notes="Failed geometric validation or insufficient inliers across all candidate boxes",
                )

            results[img_name] = best_reg_result
            images_map[img_name] = target_img
            apertures_map[img_name] = aperture

        # Save transforms.json
        from datetime import datetime, timezone

        loc_transforms = LocationTransforms(
            location_name=loc_name,
            base_image_name=base_path.name,
            base_width=w_canvas,
            base_height=h_canvas,
            created_at=datetime.now(timezone.utc).isoformat(),
            transforms=results,
        )

        if save_transforms:
            TransformStore.save(loc_path, loc_transforms)
            logger.info(f"Saved transforms to {loc_path / 'transforms.json'}")

        if self.auditor and generate_audit_images:
            self.auditor.draw_global_canvas_overview(base_canvas, results, loc_name)
            self.auditor.draw_static_composite_preview(base_canvas, results, images_map, apertures_map, loc_name)

        return loc_transforms
