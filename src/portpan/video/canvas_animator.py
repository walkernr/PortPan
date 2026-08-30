"""Canvas video animator compositing synchronized looping porthole streams onto the base panorama."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
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
        stem = image_name.split(".")[0]
        for v in video_files:
            if stem in v.name:
                return v
        return None

    def _detect_optical_aperture(self, frame: np.ndarray) -> tuple[float, float, float]:
        """Detect the circular optical aperture boundary using PolarChromaSegmenter with fallback."""
        h, w = frame.shape[:2]
        try:
            ap = self.segmenter.segment(frame)
            if ap.radius > 0.2 * min(h, w):
                return float(ap.center_x), float(ap.center_y), float(ap.radius)
        except Exception:
            pass

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < 20) / float(gray.size)

        if dark_ratio > 0.03:
            bright = (gray > 25).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
            dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist)
            cx, cy = float(max_loc[0]), float(max_loc[1])
            r = min(float(max_val) * 0.88, cy * 0.95, (h - cy) * 0.95, cx * 0.95, (w - cx) * 0.95)
            return cx, cy, r

        return float(w / 2.0), float(h / 2.0), float(min(h, w) / 2.0)

    def _create_clean_porthole_mask(
        self,
        h: int,
        w: int,
        cx: float,
        cy: float,
        r: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create clean optical porthole mask with adaptive 9% radial inset and 6% cosine taper.

        Completely eliminates contact with rubber eyecup, dark casing, or outer chromatic fringing.
        Returns (clean_mask, dist_from_center).
        """
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        inset = max(30.0, 0.09 * r)
        taper_width = max(20.0, 0.06 * r)
        r_cutoff = r - inset

        dist_inward = r_cutoff - dist_from_center
        clean_mask = np.clip(dist_inward / taper_width, 0.0, 1.0)
        clean_mask = 0.5 * (1.0 - np.cos(np.pi * clean_mask))

        return clean_mask.astype(np.float32), dist_from_center

    def _fuse_frame_roi(
        self,
        canvas_roi: np.ndarray,
        w_frame: np.ndarray,
        w_mask: np.ndarray,
        w_q: np.ndarray,
        q_curr: np.ndarray,
        ksize: int = 61,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multi-band detail fusion matching ambient low frequencies while preserving native motion sharpness.

        Uses normalized Gaussian convolution to eliminate all veiling glare, flare, and milky boundaries.
        """
        higher = (w_q > q_curr) & (w_mask > 0.01)
        if not np.any(higher):
            return canvas_roi, q_curr

        c_roi = canvas_roi.astype(np.float32)
        w_tgt = w_frame.astype(np.float32)

        tgt_low_num = cv2.GaussianBlur(w_tgt * w_mask[:, :, np.newaxis], (ksize, ksize), 0)
        tgt_low_den = cv2.GaussianBlur(w_mask, (ksize, ksize), 0)[:, :, np.newaxis] + 1e-5
        tgt_low = tgt_low_num / tgt_low_den

        canvas_low = cv2.GaussianBlur(c_roi, (ksize, ksize), 0)
        matched_tgt = canvas_low + (w_tgt - tgt_low)

        alpha = np.zeros_like(w_q)
        alpha[higher] = (
            np.clip(
                0.6 + (w_q[higher] - q_curr[higher]) / (2.0 * np.maximum(w_q[higher], 1e-4)),
                0.6,
                1.0,
            )
            * w_mask[higher]
        )
        alpha_3d = alpha[:, :, np.newaxis]

        blended = c_roi * (1.0 - alpha_3d) + matched_tgt * alpha_3d
        new_q = np.maximum(q_curr, w_q)

        return np.clip(blended, 0, 255).astype(np.uint8), new_q

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
            base_canvas_scaled = cv2.resize(base_canvas, (w_canvas, h_canvas), interpolation=cv2.INTER_LANCZOS4)
            S_canvas = np.array([[canvas_scale, 0.0, 0.0], [0.0, canvas_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            base_canvas_scaled = base_canvas.copy()
            S_canvas = np.eye(3, dtype=np.float64)

        del base_canvas
        gc.collect()

        # 3. Stream-by-stream loop preparation (Pass 1 - Memory Bounded)
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov"))
        logger.info(f"Found {len(video_files)} video streams in {videos_dir}")

        temp_dir = tempfile.TemporaryDirectory()
        temp_p = Path(temp_dir.name)

        active_streams: list[dict[str, Any]] = []

        for stream_idx, (img_name, res) in enumerate(loc_transforms.transforms.items()):
            if res.status == "FAILED" or res.transform_matrix is None:
                continue

            img_path = images_dir / img_name
            if not img_path.exists():
                continue

            H_global = res.get_matrix_np()
            matched_video = self.find_matching_video(img_name, video_files)

            if matched_video is not None:
                logger.info(f"Preparing video stream: {matched_video.name} for {img_name}")
                raw_frames = self.extractor.extract_frames(matched_video)
                if not raw_frames:
                    continue

                if res.video_to_still_transform is not None:
                    H_v2s = res.get_video_matrix_np()
                else:
                    still_img = cv2.imread(str(img_path))
                    H_v2s = self.offset_estimator.estimate_offset(still_img, matched_video)
                    del still_img

                H_video_to_canvas = S_canvas @ H_global @ H_v2s
                h_v, w_v = raw_frames[0].shape[:2]

                cx_v, cy_v, r_v = self._detect_optical_aperture(raw_frames[0])
                mask_clean, d_center = self._create_clean_porthole_mask(h_v, w_v, cx_v, cy_v, r_v)

                # Stabilize sequence to anchor frame 0
                vid_mask_f = (d_center <= r_v).astype(np.float32)
                stabilized_bgr = self.frame_stabilizer.stabilize_sequence(raw_frames, mask=vid_mask_f, anchor_index=0)
                del raw_frames

                # Synthesize seamless temporal loop
                loop_frames_bgr = self.loop_synthesizer.synthesize_loop(stabilized_bgr, target_length=loop_frames)

                # Cache stabilized loop to disk file (streaming memory bounding)
                temp_video_path = temp_p / f"stream_{stream_idx}.mp4"
                writer_temp = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w_v, h_v))
                for lf in loop_frames_bgr:
                    writer_temp.write(lf)
                writer_temp.release()

                # Saliency and quality weight
                gray_anchor = cv2.cvtColor(stabilized_bgr[0], cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray_anchor, cv2.CV_32F)
                saliency = float(np.var(lap[mask_clean > 0.5])) if np.any(mask_clean > 0.5) else 10.0
                centrality = np.clip(1.0 - (d_center / (0.82 * r_v + 1e-5)) ** 2, 0.0, 1.0)
                zoom_factor = float(np.sqrt(abs(H_video_to_canvas[0, 0] * H_video_to_canvas[1, 1] - H_video_to_canvas[0, 1] * H_video_to_canvas[1, 0])))
                saliency_boost = 1.0 + 0.6 * np.log10(1.0 + saliency)
                q_native = mask_clean * centrality * (zoom_factor ** 1.3) * saliency_boost
                saliency_score = float(np.mean(q_native[mask_clean > 0.5])) if np.any(mask_clean > 0.5) else 1.0

                del stabilized_bgr, loop_frames_bgr, gray_anchor, lap
                gc.collect()

                active_streams.append({
                    "name": img_name,
                    "is_video": True,
                    "video_path": temp_video_path,
                    "H_canvas": H_video_to_canvas,
                    "mask_native": mask_clean,
                    "q_native": q_native,
                    "saliency_score": saliency_score,
                    "src_shape": (h_v, w_v),
                })

            else:
                logger.info(f"Preparing static porthole stream for still image: {img_name}")
                still_img = cv2.imread(str(img_path))
                if still_img is None:
                    continue

                H_still_to_canvas = S_canvas @ H_global
                h_s, w_s = still_img.shape[:2]

                cx_s, cy_s, r_s = self._detect_optical_aperture(still_img)
                mask_clean, d_center = self._create_clean_porthole_mask(h_s, w_s, cx_s, cy_s, r_s)

                gray_anchor = cv2.cvtColor(still_img, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray_anchor, cv2.CV_32F)
                saliency = float(np.var(lap[mask_clean > 0.5])) if np.any(mask_clean > 0.5) else 10.0
                centrality = np.clip(1.0 - (d_center / (0.82 * r_s + 1e-5)) ** 2, 0.0, 1.0)
                zoom_factor = float(np.sqrt(abs(H_still_to_canvas[0, 0] * H_still_to_canvas[1, 1] - H_still_to_canvas[0, 1] * H_still_to_canvas[1, 0])))
                saliency_boost = 1.0 + 0.6 * np.log10(1.0 + saliency)
                q_native = mask_clean * centrality * (zoom_factor ** 1.3) * saliency_boost
                saliency_score = float(np.mean(q_native[mask_clean > 0.5])) if np.any(mask_clean > 0.5) else 1.0
                del gray_anchor, lap

                active_streams.append({
                    "name": img_name,
                    "is_video": False,
                    "image": still_img,
                    "H_canvas": H_still_to_canvas,
                    "mask_native": mask_clean,
                    "q_native": q_native,
                    "saliency_score": saliency_score,
                    "src_shape": (h_s, w_s),
                })

        if not active_streams:
            temp_dir.cleanup()
            raise PortPanError(f"No valid streams could be prepared for location: {loc_name}")

        # Sort streams by saliency score ascending: background / lower-detail laid down first,
        # highest-saliency foreground landmarks and dynamic waves laid down on top
        active_streams.sort(key=lambda s: s["saliency_score"])
        logger.info(f"Compositing order (back→front): {[s['name'] for s in active_streams]}")

        # Pre-compute bounded ROI matrices for all active streams
        for stream in active_streams:
            h_s, w_s = stream["src_shape"]
            H_c = stream["H_canvas"]
            corners = np.array([
                [0.0, 0.0],
                [w_s, 0.0],
                [w_s, h_s],
                [0.0, h_s],
            ], dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, H_c).reshape(-1, 2)
            x_min = int(max(0, np.floor(np.min(proj[:, 0]))))
            x_max = int(min(w_canvas, np.ceil(np.max(proj[:, 0]))))
            y_min = int(max(0, np.floor(np.min(proj[:, 1]))))
            y_max = int(min(h_canvas, np.ceil(np.max(proj[:, 1]))))

            if x_max > x_min and y_max > y_min:
                w_roi = x_max - x_min
                h_roi = y_max - y_min
                T_offset = np.array([
                    [1.0, 0.0, -x_min],
                    [0.0, 1.0, -y_min],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                H_roi = T_offset @ H_c
                w_mask_roi = cv2.warpPerspective(
                    stream["mask_native"],
                    H_roi,
                    (w_roi, h_roi),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                w_q_roi = cv2.warpPerspective(
                    stream["q_native"],
                    H_roi,
                    (w_roi, h_roi),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                stream["roi"] = (x_min, x_max, y_min, y_max, w_roi, h_roi, H_roi, w_mask_roi, w_q_roi)
            else:
                stream["roi"] = None

        # 4. Sequential Frame Compositing & Direct Pipe Streaming (Pass 2 - Low-RAM)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

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

        # Open video capture handles for all cached temporary video streams
        video_caps: dict[str, cv2.VideoCapture] = {}
        for s in active_streams:
            if s["is_video"]:
                video_caps[s["name"]] = cv2.VideoCapture(str(s["video_path"]))

        logger.info(f"Rendering {loop_frames} synchronized frames with {len(active_streams)} active portholes...")

        try:
            for t in range(loop_frames):
                frame_canvas = base_canvas_scaled.copy()
                priority_map = np.zeros((h_canvas, w_canvas), dtype=np.float32)

                for stream in active_streams:
                    if stream["roi"] is None:
                        continue

                    x_min, x_max, y_min, y_max, w_roi, h_roi, H_roi, w_mask_roi, w_q_roi = stream["roi"]

                    if stream["is_video"]:
                        cap = video_caps[stream["name"]]
                        ret, bgr_frame = cap.read()
                        if not ret or bgr_frame is None:
                            continue
                    else:
                        bgr_frame = stream["image"]

                    # Warp frame into bounded ROI only
                    w_bgr_roi = cv2.warpPerspective(
                        bgr_frame,
                        H_roi,
                        (w_roi, h_roi),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )

                    # Multi-band detail fusion inside ROI
                    c_roi = frame_canvas[y_min:y_max, x_min:x_max]
                    q_curr = priority_map[y_min:y_max, x_min:x_max]
                    blended_roi, new_q = self._fuse_frame_roi(
                        canvas_roi=c_roi,
                        w_frame=w_bgr_roi,
                        w_mask=w_mask_roi,
                        w_q=w_q_roi,
                        q_curr=q_curr,
                        ksize=61,
                    )
                    frame_canvas[y_min:y_max, x_min:x_max] = blended_roi
                    priority_map[y_min:y_max, x_min:x_max] = new_q

                # Stream completed frame directly into video encoder
                if proc is not None and proc.stdin is not None:
                    proc.stdin.write(frame_canvas.tobytes())
                elif writer is not None:
                    writer.write(frame_canvas)

                del frame_canvas, priority_map
                if t % 15 == 0:
                    gc.collect()

        finally:
            for cap in video_caps.values():
                cap.release()

            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
                proc.wait()
            elif writer is not None:
                writer.release()

            temp_dir.cleanup()

        logger.info(f"Successfully exported animated canvas video: {out_p}")
        return out_p
