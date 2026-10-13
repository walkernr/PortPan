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

    def deduplicate_registrations(
        self,
        transforms: dict[str, Any],
        video_files: list[Path],
        S_canvas: np.ndarray,
        cluster_distance_threshold: float = 80.0,
    ) -> list[dict[str, Any]]:
        """Cluster overlapping portholes by canvas position and pick the single best stream per cluster."""
        entries: list[dict[str, Any]] = []
        for img_name, res in transforms.items():
            if res.status == "FAILED" or res.transform_matrix is None:
                continue
            H = S_canvas @ res.get_matrix_np()
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

        logger.info(f"Deduplicated {len(entries)} registered portholes into {len(best_entries)} distinct optimal streams.")
        return best_entries

    def _detect_optical_aperture(self, frame: np.ndarray) -> tuple[float, float, float]:
        """Detect the circular optical aperture boundary using PolarChromaSegmenter with fallback."""
        h, w = frame.shape[:2]
        try:
            ap = self.segmenter.segment(frame)
            if 0.2 * min(h, w) < ap.radius < 0.6 * max(h, w):
                return ap.center_x, ap.center_y, ap.radius
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
            r = min(max_val * 0.85, cy * 0.92, (h - cy) * 0.92, cx * 0.92, (w - cx) * 0.92)
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
        """Create clean optical porthole mask with adaptive 13% radial inset and 8% cosine taper.

        Completely eliminates all physical contact with rubber eyecups, dark casing, and outer chromatic aberration arcs.
        Returns (clean_mask, dist_from_center).
        """
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        inset = max(35.0, 0.13 * r)
        taper_width = max(25.0, 0.08 * r)
        r_cutoff = r - inset

        dist_inward = r_cutoff - dist_from_center
        clean_mask = np.clip(dist_inward / taper_width, 0.0, 1.0)
        clean_mask = 0.5 * (1.0 - np.cos(np.pi * clean_mask))

        return clean_mask.astype(np.float32), dist_from_center

    def _load_isolated_mask_if_available(
        self,
        location_dir: Path,
        img_name: str,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load pre-segmented isolated porthole mask if already exported, matching static panorama exactly."""
        stem = img_name.split(".")[0]
        loc_name = location_dir.name
        iso_candidates = [
            Path("output") / loc_name / "isolated_portholes" / f"{loc_name}_{stem}_porthole.png",
            Path("output") / loc_name / "isolated_portholes" / f"{stem}_porthole.png",
        ]
        for iso_p in iso_candidates:
            if iso_p.exists():
                rgba = cv2.imread(str(iso_p), cv2.IMREAD_UNCHANGED)
                if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                    if rgba.shape[0] != h or rgba.shape[1] != w:
                        rgba = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA)
                    alpha_orig = rgba[:, :, 3].astype(np.float32)
                    dist = cv2.distanceTransform((alpha_orig > 128).astype(np.uint8), cv2.DIST_L2, 5)
                    r_max = float(dist.max()) if dist.size > 0 else 1000.0
                    inset = max(35.0, 0.13 * r_max)
                    taper_width = max(25.0, 0.08 * r_max)
                    clean_mask = np.clip((dist - inset) / taper_width, 0.0, 1.0)
                    clean_mask = (0.5 * (1.0 - np.cos(np.pi * clean_mask))).astype(np.float32)
                    d_center = r_max - dist
                    return clean_mask, d_center
        return None

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
        max_dimension: int | None = 3840,
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

        # Standard video compatibility: Cap maximum dimension to 4K UHD bounds (e.g. 3840) to prevent
        # GPU hardware decoder failures and ensure universal desktop/browser playback
        if max_dimension and (w_canvas > max_dimension or h_canvas > max_dimension):
            scale_down = min(float(max_dimension) / float(w_canvas), float(max_dimension) / float(h_canvas))
            w_canvas = round(w_canvas * scale_down)
            h_canvas = round(h_canvas * scale_down)

        # Ensure even dimensions for H.264 / yuv420p
        w_canvas = w_canvas & ~1
        h_canvas = h_canvas & ~1

        sx = float(w_canvas) / float(w_orig)
        sy = float(h_canvas) / float(h_orig)
        logger.info(f"Target video canvas dimensions: {w_canvas}x{h_canvas} (scale: {sx:.4f}x, {sy:.4f}y)")

        base_canvas_scaled = cv2.resize(base_canvas, (w_canvas, h_canvas), interpolation=cv2.INTER_LANCZOS4)
        S_canvas = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        del base_canvas
        gc.collect()

        # 3. Deduplicate registrations to eliminate redundant overlapping bubbles
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov"))
        logger.info(f"Found {len(video_files)} video files in {videos_dir}")

        best_entries = self.deduplicate_registrations(
            loc_transforms.transforms,
            video_files,
            S_canvas=S_canvas,
            cluster_distance_threshold=80.0,
        )

        temp_dir = tempfile.TemporaryDirectory()
        temp_p = Path(temp_dir.name)

        active_streams: list[dict[str, Any]] = []

        # 4. Stream-by-stream loop preparation (Pass 1 - Strictly Memory Bounded)
        for stream_idx, entry in enumerate(best_entries):
            img_name = entry["img_name"]
            res = entry["res"]
            matched_video = entry["matched_video"]

            img_path = images_dir / img_name
            if not img_path.exists():
                continue

            H_global = res.get_matrix_np()

            if matched_video is not None:
                logger.info(f"Preparing video stream {stream_idx+1}/{len(best_entries)}: {matched_video.name}")
                raw_frames = self.extractor.extract_frames(matched_video)
                if not raw_frames:
                    continue

                if res.video_to_still_transform is not None:
                    H_v2s = res.get_video_matrix_np()
                else:
                    still_img = cv2.imread(str(img_path))
                    if still_img is not None:
                        H_v2s = self.offset_estimator.estimate_offset(still_img, matched_video)
                        del still_img
                    else:
                        H_v2s = np.eye(3, dtype=np.float64)

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
                centrality = np.clip(1.0 - (d_center / (0.80 * r_v + 1e-5)) ** 2, 0.0, 1.0)
                zoom_factor = float(np.sqrt(abs(H_video_to_canvas[0, 0] * H_video_to_canvas[1, 1] - H_video_to_canvas[0, 1] * H_video_to_canvas[1, 0])))
                saliency_boost = 1.0 + 0.6 * np.log10(1.0 + saliency)
                q_native = mask_clean * centrality * (zoom_factor ** 1.3) * saliency_boost
                footprint_area = np.pi * ((r_v * zoom_factor) ** 2)
                sort_metric = footprint_area / (saliency + 1e-2)

                del stabilized_bgr, loop_frames_bgr, gray_anchor, lap

                # Precompute bounded ROI immediately to avoid storing full-resolution native masks in RAM
                corners = np.array([[0.0, 0.0], [w_v, 0.0], [w_v, h_v], [0.0, h_v]], dtype=np.float32).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(corners, H_video_to_canvas).reshape(-1, 2)
                x_min = max(0, int(np.floor(np.min(proj[:, 0]))))
                x_max = min(w_canvas, int(np.ceil(np.max(proj[:, 0]))))
                y_min = max(0, int(np.floor(np.min(proj[:, 1]))))
                y_max = min(h_canvas, int(np.ceil(np.max(proj[:, 1]))))

                if x_max > x_min and y_max > y_min:
                    w_roi = x_max - x_min
                    h_roi = y_max - y_min
                    T_offset = np.array([[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]], dtype=np.float64)
                    H_roi = T_offset @ H_video_to_canvas
                    w_mask_roi = cv2.warpPerspective(
                        mask_clean,
                        H_roi,
                        (w_roi, h_roi),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    )
                    w_q_roi = cv2.warpPerspective(
                        q_native,
                        H_roi,
                        (w_roi, h_roi),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    )

                    active_streams.append({
                        "name": img_name,
                        "is_video": True,
                        "video_path": temp_video_path,
                        "H_roi": H_roi,
                        "roi": (x_min, x_max, y_min, y_max, w_roi, h_roi),
                        "w_mask_roi": w_mask_roi,
                        "w_q_roi": w_q_roi,
                        "sort_metric": sort_metric,
                        "inliers": res.inlier_count,
                    })

                del mask_clean, q_native
                gc.collect()

            else:
                logger.info(f"Preparing static stream {stream_idx+1}/{len(best_entries)}: {img_name}")
                still_img = cv2.imread(str(img_path))
                if still_img is None:
                    continue

                H_still_to_canvas = S_canvas @ H_global
                h_s, w_s = still_img.shape[:2]

                # Check if an isolated porthole mask was already computed and exported
                iso_result = self._load_isolated_mask_if_available(loc_path, img_name, h_s, w_s)
                if iso_result is not None:
                    mask_clean, d_center = iso_result
                    r_s = float(w_s / 2.0)
                else:
                    cx_s, cy_s, r_s = self._detect_optical_aperture(still_img)
                    mask_clean, d_center = self._create_clean_porthole_mask(h_s, w_s, cx_s, cy_s, r_s)

                gray_anchor = cv2.cvtColor(still_img, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray_anchor, cv2.CV_32F)
                saliency = float(np.var(lap[mask_clean > 0.5])) if np.any(mask_clean > 0.5) else 10.0
                centrality = np.clip(1.0 - (d_center / (0.80 * r_s + 1e-5)) ** 2, 0.0, 1.0)
                zoom_factor = float(np.sqrt(abs(H_still_to_canvas[0, 0] * H_still_to_canvas[1, 1] - H_still_to_canvas[0, 1] * H_still_to_canvas[1, 0])))
                saliency_boost = 1.0 + 0.6 * np.log10(1.0 + saliency)
                q_native = mask_clean * centrality * (zoom_factor ** 1.3) * saliency_boost
                footprint_area = np.pi * ((r_s * zoom_factor) ** 2)
                sort_metric = footprint_area / (saliency + 1e-2)
                del gray_anchor, lap

                # Precompute bounded ROI and pre-warp static still frame into ROI once
                corners = np.array([[0.0, 0.0], [w_s, 0.0], [w_s, h_s], [0.0, h_s]], dtype=np.float32).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(corners, H_still_to_canvas).reshape(-1, 2)
                x_min = max(0, int(np.floor(np.min(proj[:, 0]))))
                x_max = min(w_canvas, int(np.ceil(np.max(proj[:, 0]))))
                y_min = max(0, int(np.floor(np.min(proj[:, 1]))))
                y_max = min(h_canvas, int(np.ceil(np.max(proj[:, 1]))))

                if x_max > x_min and y_max > y_min:
                    w_roi = x_max - x_min
                    h_roi = y_max - y_min
                    T_offset = np.array([[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]], dtype=np.float64)
                    H_roi = T_offset @ H_still_to_canvas
                    w_mask_roi = cv2.warpPerspective(
                        mask_clean,
                        H_roi,
                        (w_roi, h_roi),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    )
                    w_q_roi = cv2.warpPerspective(
                        q_native,
                        H_roi,
                        (w_roi, h_roi),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    )
                    w_bgr_roi = cv2.warpPerspective(
                        still_img,
                        H_roi,
                        (w_roi, h_roi),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )

                    active_streams.append({
                        "name": img_name,
                        "is_video": False,
                        "w_bgr_roi": w_bgr_roi,
                        "H_roi": H_roi,
                        "roi": (x_min, x_max, y_min, y_max, w_roi, h_roi),
                        "w_mask_roi": w_mask_roi,
                        "w_q_roi": w_q_roi,
                        "sort_metric": sort_metric,
                        "inliers": res.inlier_count,
                    })

                del still_img, mask_clean, q_native
                gc.collect()

        if not active_streams:
            temp_dir.cleanup()
            raise PortPanError(f"No valid streams could be prepared for location: {loc_name}")

        # Sort streams by footprint_area / saliency descending (broad/low-detail first, sharp landmarks on top)
        active_streams.sort(key=lambda s: (s["sort_metric"], -s["inliers"]), reverse=True)
        logger.info(f"Compositing order (back→front): {[s['name'] for s in active_streams]}")

        # 5. Sequential Frame Compositing & Direct Pipe Streaming (Pass 2 - Strictly Low-RAM)
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
                    x_min, x_max, y_min, y_max, w_roi, h_roi = stream["roi"]
                    w_mask_roi = stream["w_mask_roi"]
                    w_q_roi = stream["w_q_roi"]
                    H_roi = stream["H_roi"]

                    if stream["is_video"]:
                        cap = video_caps[stream["name"]]
                        ret, bgr_frame = cap.read()
                        if not ret or bgr_frame is None:
                            continue
                        w_bgr_roi = cv2.warpPerspective(
                            bgr_frame,
                            H_roi,
                            (w_roi, h_roi),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0),
                        )
                    else:
                        w_bgr_roi = stream["w_bgr_roi"]

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
