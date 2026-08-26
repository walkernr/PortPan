"""Canvas video animator compositing synchronized looping porthole streams onto the base panorama."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from portpan.core.exceptions import PortPanError
from portpan.core.models import CircleAperture, LocationTransforms, RegistrationResult  # noqa: F401
from portpan.geometry.transform_store import TransformStore
from portpan.segmentation.base import BaseApertureSegmenter
from portpan.segmentation.polar_segmenter import PolarChromaSegmenter
from portpan.video.frame_stabilizer import VideoFrameStabilizer
from portpan.video.loop_synthesizer import LoopMode, SeamlessLoopSynthesizer
from portpan.video.stabilizer_offset import VideoStabilizerOffsetEstimator
from portpan.video.stream_extractor import VideoStreamExtractor

logger = logging.getLogger("portpan.video")


class CanvasVideoAnimator:
    """Composites multiple synchronized Live Photo video streams onto a master panoramic canvas."""

    def __init__(
        self,
        extractor: VideoStreamExtractor | None = None,
        loop_synthesizer: SeamlessLoopSynthesizer | None = None,
        offset_estimator: VideoStabilizerOffsetEstimator | None = None,
        segmenter: BaseApertureSegmenter | None = None,
        frame_stabilizer: VideoFrameStabilizer | None = None,
    ) -> None:
        self.extractor = extractor or VideoStreamExtractor()
        self.loop_synthesizer = loop_synthesizer or SeamlessLoopSynthesizer(mode=LoopMode.CROSSFADE)
        self.offset_estimator = offset_estimator or VideoStabilizerOffsetEstimator()
        self.segmenter = segmenter or PolarChromaSegmenter()
        self.frame_stabilizer = frame_stabilizer or VideoFrameStabilizer()

    def find_matching_video(self, image_name: str, video_files: list[Path]) -> Path | None:
        """Find the matching MP4 Live Photo video file for a given still image."""
        # e.g. PXL_20260823_215141677.MP.jpg -> PXL_20260823_215141677
        stem = image_name.split(".")[0]
        for v in video_files:
            if stem in v.name:
                return v
        return None

    def animate_location(
        self,
        location_dir: Path | str,
        output_path: Path | str,
        loop_frames: int = 60,
        fps: float = 30.0,
        canvas_scale: float = 1.5,
    ) -> Path:
        """Generate a synchronized, seamlessly looping video master of all registered portholes on the base canvas."""
        loc_path = Path(location_dir)
        loc_name = loc_path.name
        base_dir = loc_path / "base"
        images_dir = loc_path / "images"
        videos_dir = loc_path / "videos"
        transforms_file = loc_path / "transforms.json"

        # 1. Load transforms
        if not transforms_file.exists():
            raise PortPanError(f"Transforms file not found: {transforms_file}. Run registration first.")

        store = TransformStore()
        loc_transforms = store.load(loc_path)
        if loc_transforms is None:
            raise PortPanError(f"Failed to load transforms from {loc_path}")

        # 2. Load master base canvas
        base_files = list(base_dir.glob("*.jpg")) + list(base_dir.glob("*.png"))
        if not base_files:
            raise PortPanError(f"No base panorama image found in {base_dir}")
        base_canvas = cv2.imread(str(base_files[0]))
        if base_canvas is None:
            raise PortPanError(f"Failed to read base image: {base_files[0]}")

        h_orig, w_orig = base_canvas.shape[:2]
        w_canvas = int(round(w_orig * canvas_scale))
        h_canvas = int(round(h_orig * canvas_scale))

        if canvas_scale != 1.0:
            base_canvas_scaled = cv2.resize(base_canvas, (w_canvas, h_canvas), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
            S_canvas = np.array([[canvas_scale, 0.0, 0.0], [0.0, canvas_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            base_canvas_scaled = base_canvas.copy().astype(np.float32)
            S_canvas = np.eye(3, dtype=np.float64)

        # 3. Find and prepare all video streams
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov"))
        logger.info(f"Found {len(video_files)} video streams in {videos_dir}")

        active_streams: list[dict[str, Any]] = []

        for img_name, res in loc_transforms.transforms.items():
            if res.status == "FAILED":
                continue

            img_path = images_dir / img_name
            if not img_path.exists():
                continue

            matched_video = self.find_matching_video(img_name, video_files)
            if not matched_video:
                logger.debug(f"No video stream found for {img_name}, using static image")
                continue

            logger.info(f"Preparing video stream: {matched_video.name} for {img_name}")
            raw_frames = self.extractor.extract_frames(matched_video)

            # Load still image for aperture and EIS offset estimation
            still_img = cv2.imread(str(img_path))
            if still_img is None:
                continue

            # Detect optical aperture
            aperture = self.segmenter.segment(still_img)

            # Estimate or retrieve video-to-still matrix
            if res.video_to_still_transform is not None:
                H_v2s = res.get_video_matrix_np()
            else:
                H_v2s = self.offset_estimator.estimate_offset(still_img, matched_video)

            H_global = res.get_matrix_np()
            # Full composite matrix mapping video pixel space -> master canvas space
            H_video_to_canvas = S_canvas @ H_global @ H_v2s

            # Detect optical aperture directly on the video keyframe for pixel-perfect mask alignment
            vid_aperture = self.segmenter.segment(raw_frames[0])
            vid_mask_u8 = vid_aperture.generate_mask(feather_px=2.0)
            vid_mask_f = vid_mask_u8.astype(np.float32) / 255.0

            # Stabilize raw BGR video frames to lock stationary optical features to anchor keyframe
            logger.info(f"Stabilizing {len(raw_frames)} video frames for {img_name} to anchor keyframe...")
            stabilized_bgr = self.frame_stabilizer.stabilize_sequence(raw_frames, mask=vid_mask_f, anchor_index=0)

            # Synthesize seamlessly looping sequence from stabilized BGR frames
            loop_stream_bgr = self.loop_synthesizer.synthesize_loop(stabilized_bgr, target_length=loop_frames)

            # Precompute per-frame dynamic aperture masks with soft boundary feathering
            logger.info(f"Computing frame-wise dynamic aperture masks for {img_name} ({loop_frames} frames)...")
            warped_canvas_masks: list[np.ndarray] = []
            for t_idx, f in enumerate(loop_stream_bgr):
                ap_t = self.segmenter.segment(f)
                
                # Valid illumination gating (eliminates any black warp/stabilization borders)
                g_f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                b_f, _, r_f = cv2.split(f.astype(np.float32))
                valid_f = ((g_f > 35) & (r_f - b_f < 12.0)).astype(np.float32)
                valid_f = cv2.erode(valid_f, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
                
                mask_t = np.zeros(f.shape[:2], dtype=np.float32)
                r_draw = max(1.0, ap_t.radius - 2.0)
                cv2.circle(mask_t, (int(round(ap_t.center_x)), int(round(ap_t.center_y))), int(round(r_draw)), 1.0, -1)
                mask_t = mask_t * valid_f
                # Soft cosine / Gaussian feather at dispersion boundary
                mask_t = cv2.GaussianBlur(mask_t, (21, 21), 6.0)
                
                # Warp to master canvas
                w_mask = cv2.warpPerspective(
                    mask_t,
                    H_video_to_canvas,
                    (w_canvas, h_canvas),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                w_mask = np.clip(w_mask, 0.0, 1.0)[:, :, np.newaxis]
                warped_canvas_masks.append(w_mask)

            active_streams.append({
                "name": img_name,
                "frames": loop_stream_bgr,
                "H_video_to_canvas": H_video_to_canvas,
                "warped_masks": warped_canvas_masks,
                "footprint_area": float(np.sum(warped_canvas_masks[0])),
            })

        if not active_streams:
            raise PortPanError(f"No valid video streams could be prepared for location: {loc_name}")

        # Sort by warped footprint area descending: largest (most zoomed-out) behind, smallest on top
        active_streams.sort(key=lambda s: s["footprint_area"], reverse=True)
        logger.info(f"Z-order (back→front): {[s['name'] for s in active_streams]}")

        logger.info(f"Rendering {loop_frames} synchronized frames with {len(active_streams)} active portholes...")

        # 4. Render synchronized frames and encode video
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w_canvas, h_canvas))
        if not writer.isOpened():
            # Fallback to MJPG / XVID if mp4v is not supported by backend
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w_canvas, h_canvas))

        for t in range(loop_frames):
            frame_canvas = base_canvas_scaled.copy()

            for stream in active_streams:
                bgr_frame = stream["frames"][t]
                H_v2c = stream["H_video_to_canvas"]
                warped_mask = stream["warped_masks"][t]

                # Warp stabilized BGR video frame directly into master canvas coordinates
                warped_bgr = cv2.warpPerspective(
                    bgr_frame,
                    H_v2c,
                    (w_canvas, h_canvas),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                ).astype(np.float32)

                # Composite over master canvas using Porter-Duff Over with dynamic optical mask
                frame_canvas = frame_canvas * (1.0 - warped_mask) + warped_bgr * warped_mask

            final_frame_uint8 = np.clip(frame_canvas, 0, 255).astype(np.uint8)
            writer.write(final_frame_uint8)

        writer.release()
        logger.info(f"Successfully exported animated canvas video: {out_p}")
        return out_p
