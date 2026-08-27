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

    def deduplicate_registrations(
        self,
        transforms: dict[str, Any],
        video_files: list[Path],
        cluster_distance_threshold: float = 180.0,
    ) -> list[dict[str, Any]]:
        """Cluster registered portholes by canvas position and pick the single best stream per cluster."""
        entries: list[dict[str, Any]] = []
        for img_name, res in transforms.items():
            if res.status == "FAILED" or res.transform_matrix is None:
                continue
            H = res.get_matrix_np()
            pt = np.array([2000.0, 1500.0, 1.0])
            w_pt = H @ pt
            cx, cy = float(w_pt[0] / w_pt[2]), float(w_pt[1] / w_pt[2])
            v_match = self.find_matching_video(img_name, video_files)
            entries.append({
                "img_name": img_name,
                "res": res,
                "cx": cx,
                "cy": cy,
                "matched_video": v_match,
                "inliers": res.inlier_count,
            })

        clusters: list[list[dict[str, Any]]] = []
        for e in entries:
            matched_c = None
            for c in clusters:
                dist = float(np.hypot(e["cx"] - c[0]["cx"], e["cy"] - c[0]["cy"]))
                if dist < cluster_distance_threshold:
                    matched_c = c
                    break
            if matched_c is not None:
                matched_c.append(e)
            else:
                clusters.append([e])

        # Pick best from each cluster: prefer video stream, then highest inlier count
        best_entries: list[dict[str, Any]] = []
        for c in clusters:
            c_sorted = sorted(c, key=lambda x: (1 if x["matched_video"] is not None else 0, x["inliers"]), reverse=True)
            best_entries.append(c_sorted[0])

        return best_entries

    def _detect_video_aperture(self, frame: np.ndarray) -> tuple[float, float, float]:
        """Inscribe an exact circle inside the illuminated optical aperture with zero dark barrel leakage and zero frame clipping."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < 20) / float(gray.size)

        # Telescope portholes have significant optical barrel (> 3% dark pixels)
        if dark_ratio > 0.03:
            bright = (gray > 25).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
            dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist)
            cx, cy = float(max_loc[0]), float(max_loc[1])
            # Inscribed radius (86%) strictly inside illuminated optical zone AND within frame boundaries
            r = min(float(max_val) * 0.86, cy * 0.94, (h - cy) * 0.94, cx * 0.94, (w - cx) * 0.94)
            return cx, cy, r

        # Wide shot without barrel fills full frame
        return float(w / 2.0), float(h / 2.0), float(min(h, w) / 2.0)

    def animate_location(
        self,
        location_dir: str | Path,
        output_path: str | Path,
        loop_frames: int = 60,
        fps: float = 30.0,
        canvas_scale: float = 1.0,
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

        # 3. Find and prepare all registered porthole streams (preserving full multi-porthole coverage)
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov"))
        logger.info(f"Found {len(video_files)} video streams in {videos_dir}")

        active_streams: list[dict[str, Any]] = []

        for img_name, res in loc_transforms.transforms.items():
            if res.status == "FAILED" or res.transform_matrix is None:
                continue

            img_path = images_dir / img_name
            if not img_path.exists():
                continue

            still_img = cv2.imread(str(img_path))
            if still_img is None:
                continue

            H_global = res.get_matrix_np()
            matched_video = self.find_matching_video(img_name, video_files)

            if matched_video is not None:
                logger.info(f"Preparing video stream: {matched_video.name} for {img_name}")
                raw_frames = self.extractor.extract_frames(matched_video)

                if res.video_to_still_transform is not None:
                    H_v2s = res.get_video_matrix_np()
                else:
                    H_v2s = self.offset_estimator.estimate_offset(still_img, matched_video)

                H_video_to_canvas = S_canvas @ H_global @ H_v2s

                # Inscribe optical aperture strictly inside illuminated zone to guarantee ZERO black barrel leakage
                cx_v, cy_v, r_v = self._detect_video_aperture(raw_frames[0])
                vid_mask_u8 = np.zeros(raw_frames[0].shape[:2], dtype=np.uint8)
                cv2.circle(vid_mask_u8, (int(round(cx_v)), int(round(cy_v))), int(round(r_v)), 255, -1)
                vid_mask_f = vid_mask_u8.astype(np.float32) / 255.0

                # Stabilize video sequence to anchor frame 0
                stabilized_bgr = self.frame_stabilizer.stabilize_sequence(raw_frames, mask=vid_mask_f, anchor_index=0)

                # Native smooth optical feathered aperture mask (12px feathering)
                Y, X = np.ogrid[: raw_frames[0].shape[0], : raw_frames[0].shape[1]]
                d_from_center = np.sqrt((X - cx_v) ** 2 + (Y - cy_v) ** 2)
                mask_native = np.clip((r_v - d_from_center) / 12.0, 0.0, 1.0).astype(np.float32)

                # Footprint area on canvas for z-ordering
                footprint_area = np.pi * ((r_v * np.sqrt(H_video_to_canvas[0, 0] ** 2 + H_video_to_canvas[0, 1] ** 2)) ** 2)

                active_streams.append({
                    "name": img_name,
                    "is_video": True,
                    "frames": stabilized_bgr,
                    "H_canvas": H_video_to_canvas,
                    "mask_native": mask_native,
                    "footprint_area": float(footprint_area),
                })

            else:
                # Check if this still image is a duplicate of an existing video stream (distance < 120px on canvas)
                H_still_to_canvas = S_canvas @ H_global
                pt_can = H_still_to_canvas @ np.array([still_img.shape[1] / 2.0, still_img.shape[0] / 2.0, 1.0])
                pt_can /= pt_can[2]

                is_duplicate = False
                for vs in active_streams:
                    if vs["is_video"]:
                        vs_c = vs["H_canvas"] @ np.array([vs["frames"][0].shape[1] / 2.0, vs["frames"][0].shape[0] / 2.0, 1.0])
                        vs_c /= vs_c[2]
                        if float(np.hypot(pt_can[0] - vs_c[0], pt_can[1] - vs_c[1])) < 120.0:
                            is_duplicate = True
                            break

                if is_duplicate:
                    logger.info(f"Skipping static duplicate {img_name} (active video stream already covers this location)")
                    continue

                # Static still-image porthole fallback with clean inscribed optical aperture
                logger.info(f"Preparing static porthole stream for still image: {img_name}")
                cx_s, cy_s, r_s = self._detect_video_aperture(still_img)
                Y_s, X_s = np.ogrid[: still_img.shape[0], : still_img.shape[1]]
                d_from_center_s = np.sqrt((X_s - cx_s) ** 2 + (Y_s - cy_s) ** 2)
                still_mask_f = np.clip((r_s - d_from_center_s) / 12.0, 0.0, 1.0).astype(np.float32)

                footprint_area = np.pi * ((r_s * np.sqrt(H_still_to_canvas[0, 0] ** 2 + H_still_to_canvas[0, 1] ** 2)) ** 2)

                active_streams.append({
                    "name": img_name,
                    "is_video": False,
                    "image": still_img,
                    "H_canvas": H_still_to_canvas,
                    "mask_native": still_mask_f,
                    "footprint_area": float(footprint_area),
                })

        if not active_streams:
            raise PortPanError(f"No valid streams could be prepared for location: {loc_name}")

        # Sort by footprint area descending: largest (zoomed out) behind, smallest/highest-zoom on top
        active_streams.sort(key=lambda s: s["footprint_area"], reverse=True)
        logger.info(f"Z-order (back→front): {[s['name'] for s in active_streams]}")

        logger.info(f"Rendering {loop_frames} synchronized frames with {len(active_streams)} active portholes (low-RAM streaming mode)...")

        # 4. Stream frame-by-frame directly to FFmpeg H.264 pipe
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        import subprocess

        ffmpeg_bin = shutil.which("ffmpeg")
        proc = None
        writer = None

        if ffmpeg_bin:
            logger.info("Encoding video via streaming FFmpeg pipe (libx264, visually lossless CRF 14)...")
            ffmpeg_cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w_canvas}x{h_canvas}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "14",
                "-pix_fmt", "yuv420p",
                str(out_p),
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        else:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w_canvas, h_canvas))

        erode_kernel = np.ones((5, 5), np.uint8)

        for t in range(loop_frames):
            frame_canvas = base_canvas_scaled.copy().astype(np.uint8)

            for stream in active_streams:
                if stream["is_video"]:
                    bgr_frame = self.loop_synthesizer.synthesize_frame(stream["frames"], t, target_length=loop_frames)
                else:
                    bgr_frame = stream["image"]

                H_mat = stream["H_canvas"]
                mask_native = stream["mask_native"]

                # Warp mask and frame to canvas
                w_mask = cv2.warpPerspective(
                    mask_native,
                    H_mat,
                    (w_canvas, h_canvas),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                w_bgr = cv2.warpPerspective(
                    bgr_frame,
                    H_mat,
                    (w_canvas, h_canvas),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

                # Composite via ROI bounding box for speed and near-zero RAM footprint
                nz_rows, nz_cols = np.where(w_mask > 0.001)
                if len(nz_rows) > 0:
                    ymin, ymax = int(np.min(nz_rows)), int(np.max(nz_rows)) + 1
                    xmin, xmax = int(np.min(nz_cols)), int(np.max(nz_cols)) + 1

                    m_roi = w_mask[ymin:ymax, xmin:xmax, np.newaxis]
                    c_roi = frame_canvas[ymin:ymax, xmin:xmax].astype(np.float32)
                    b_roi = w_bgr[ymin:ymax, xmin:xmax].astype(np.float32)

                    blended_roi = c_roi * (1.0 - m_roi) + b_roi * m_roi
                    frame_canvas[ymin:ymax, xmin:xmax] = np.clip(blended_roi, 0, 255).astype(np.uint8)

            # Output completed frame directly to encoder
            if proc is not None and proc.stdin is not None:
                proc.stdin.write(frame_canvas.tobytes())
            elif writer is not None:
                writer.write(frame_canvas)

        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
            proc.wait()
        elif writer is not None:
            writer.release()

        logger.info(f"Successfully exported animated canvas video: {out_p}")
        return out_p
