"""Video frame stabilization offset estimation between MP4 streams and still JPEG frames."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from portpan.core.exceptions import VideoProcessingError


class VideoStabilizerOffsetEstimator:
    """Estimates the 3x3 geometric transform H_(video -> still) to correct for Electronic Image Stabilization."""

    def __init__(self, keyframe_index: int = 0) -> None:
        self.keyframe_index = keyframe_index

    def extract_keyframe(self, video_path: Path | str) -> np.ndarray:
        """Extract a single BGR keyframe from the video stream."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = min(self.keyframe_index, max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise VideoProcessingError(f"Failed to read keyframe {target_frame} from {video_path}")

        return frame

    def estimate_offset(self, still_image: np.ndarray, video_path: Path | str) -> np.ndarray:
        """Compute the 3x3 homography matrix mapping video frame pixel space directly into still image pixel space."""
        keyframe = self.extract_keyframe(video_path)
        h_vid, w_vid = keyframe.shape[:2]
        h_still, w_still = still_image.shape[:2]

        # Initial scale matrix from video dimensions to still dimensions
        sx = w_still / float(w_vid)
        sy = h_still / float(h_vid)
        S_init = np.diag([sx, sy, 1.0])

        # Resize keyframe to still image dimensions for feature matching
        keyframe_scaled = cv2.resize(keyframe, (w_still, h_still), interpolation=cv2.INTER_LINEAR)

        gray_still = cv2.cvtColor(still_image, cv2.COLOR_BGR2GRAY) if len(still_image.shape) == 3 else still_image
        gray_keyframe = cv2.cvtColor(keyframe_scaled, cv2.COLOR_BGR2GRAY) if len(keyframe_scaled.shape) == 3 else keyframe_scaled

        sift = cv2.SIFT_create(nfeatures=4000)
        kp_still, des_still = sift.detectAndCompute(gray_still, None)
        kp_vid, des_vid = sift.detectAndCompute(gray_keyframe, None)

        if des_still is None or des_vid is None or len(kp_still) < 10 or len(kp_vid) < 10:
            return S_init

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_vid, des_still, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return S_init

        pts_vid = np.float32([kp_vid[m.queryIdx].pt for m in good_matches])
        pts_still = np.float32([kp_still[m.trainIdx].pt for m in good_matches])

        # Estimate partial affine / similarity transform between scaled video and still
        M_aff, mask = cv2.estimateAffinePartial2D(pts_vid, pts_still, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if M_aff is None:
            return S_init

        H_scaled_to_still = np.vstack([M_aff, [0.0, 0.0, 1.0]])

        # Global video-to-still matrix = H_scaled_to_still * S_init
        H_video_to_still = H_scaled_to_still @ S_init
        return H_video_to_still
