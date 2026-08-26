"""Video frame stabilization and optical feature anchoring for digiscope video streams."""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger("portpan.video.stabilizer")


class VideoFrameStabilizer:
    """Stabilizes video frame sequences to a reference anchor frame using robust optical flow tracking.

    Anchors stationary scene features (rocks, terrain, landmarks) across all frames of a
    Live Photo stream to eliminate handheld drift and shake, while preserving dynamic motion
    (waves, ripples, swaying foliage) inside the scene.
    """

    def __init__(
        self,
        max_features: int = 500,
        quality_level: float = 0.01,
        min_distance: float = 10.0,
        ransac_threshold: float = 2.0,
    ) -> None:
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_threshold = ransac_threshold

    def stabilize_sequence(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray | None = None,
        anchor_index: int = 0,
    ) -> list[np.ndarray]:
        """Stabilize a sequence of video frames to lock all frames to the anchor keyframe.

        Args:
            frames: List of input BGR video frames.
            mask: Optional binary or float alpha mask (0.0-1.0 or 0-255) defining the
                optical aperture region to track features within.
            anchor_index: Index of the reference anchor frame (default 0).

        Returns:
            List of stabilized BGR video frames aligned to the anchor frame.
        """
        if not frames:
            return []
        if len(frames) == 1:
            return [frames[0].copy()]

        h, w = frames[0].shape[:2]

        # Prepare tracking mask (erode slightly so we don't track the moving aperture edge itself)
        track_mask = None
        if mask is not None:
            if mask.dtype == np.float32 or mask.dtype == np.float64:
                mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                mask_u8 = mask.copy()
            # Erode mask by 15px so edge fringe/rim doesn't bias optical scene tracking
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            track_mask = cv2.erode((mask_u8 > 128).astype(np.uint8) * 255, kernel)

        # Convert frames to grayscale
        gray_frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f for f in frames
        ]

        # Detect good features to track on anchor frame
        anchor_gray = gray_frames[anchor_index]
        p0 = cv2.goodFeaturesToTrack(
            anchor_gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=track_mask,
        )

        if p0 is None or len(p0) < 6:
            logger.warning(
                f"Too few trackable features ({0 if p0 is None else len(p0)}) for stabilization; returning original frames"
            )
            return [f.copy() for f in frames]

        # Compute cumulative 3x3 homography / affine from each frame t to anchor (index 0)
        num_frames = len(frames)
        transforms_to_anchor: list[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in range(num_frames)]

        # Forward pass from anchor_index to end
        curr_pts = p0.copy()
        curr_H = np.eye(3, dtype=np.float64)

        for t in range(anchor_index + 1, num_frames):
            prev_g = gray_frames[t - 1]
            curr_g = gray_frames[t]

            # Replenish features if point count has dropped
            if curr_pts is None or len(curr_pts) < 30:
                new_pts = cv2.goodFeaturesToTrack(
                    prev_g,
                    maxCorners=self.max_features,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    mask=track_mask,
                )
                if new_pts is not None and len(new_pts) >= 6:
                    curr_pts = new_pts

            if curr_pts is None or len(curr_pts) < 6:
                transforms_to_anchor[t] = curr_H.copy()
                continue

            # Forward LK tracking
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_g,
                curr_g,
                curr_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            # Backward LK tracking for forward-backward consistency check
            back_pts, status_b, err_b = cv2.calcOpticalFlowPyrLK(
                curr_g,
                prev_g,
                next_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            d = np.abs(curr_pts - back_pts).reshape(-1, 2).max(axis=1)
            good = (status.ravel() == 1) & (status_b.ravel() == 1) & (d < 1.0)

            pts_prev_good = curr_pts[good]
            pts_curr_good = next_pts[good]

            if len(pts_curr_good) >= 6:
                M_step, inliers = cv2.estimateAffinePartial2D(
                    pts_curr_good,
                    pts_prev_good,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold,
                )
                if M_step is not None:
                    H_step = np.vstack([M_step, [0.0, 0.0, 1.0]])
                    curr_H = curr_H @ H_step
                    # Update active points to inliers
                    if inliers is not None:
                        inlier_mask = inliers.ravel() == 1
                        curr_pts = pts_curr_good[inlier_mask].reshape(-1, 1, 2)
                    else:
                        curr_pts = pts_curr_good.reshape(-1, 1, 2)
                else:
                    curr_pts = pts_curr_good.reshape(-1, 1, 2)
            else:
                curr_pts = None

            transforms_to_anchor[t] = curr_H.copy()

        # Backward pass from anchor_index to 0 (if anchor_index > 0)
        if anchor_index > 0:
            curr_pts = p0.copy()
            curr_H = np.eye(3, dtype=np.float64)
            for t in range(anchor_index - 1, -1, -1):
                prev_g = gray_frames[t + 1]
                curr_g = gray_frames[t]

                if curr_pts is None or len(curr_pts) < 30:
                    new_pts = cv2.goodFeaturesToTrack(
                        prev_g,
                        maxCorners=self.max_features,
                        qualityLevel=self.quality_level,
                        minDistance=self.min_distance,
                        mask=track_mask,
                    )
                    if new_pts is not None and len(new_pts) >= 6:
                        curr_pts = new_pts

                if curr_pts is None or len(curr_pts) < 6:
                    transforms_to_anchor[t] = curr_H.copy()
                    continue

                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_g,
                    curr_g,
                    curr_pts,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                back_pts, status_b, err_b = cv2.calcOpticalFlowPyrLK(
                    curr_g,
                    prev_g,
                    next_pts,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )

                d = np.abs(curr_pts - back_pts).reshape(-1, 2).max(axis=1)
                good = (status.ravel() == 1) & (status_b.ravel() == 1) & (d < 1.0)
                pts_prev_good = curr_pts[good]
                pts_curr_good = next_pts[good]

                if len(pts_curr_good) >= 6:
                    M_step, inliers = cv2.estimateAffinePartial2D(
                        pts_curr_good,
                        pts_prev_good,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=self.ransac_threshold,
                    )
                    if M_step is not None:
                        H_step = np.vstack([M_step, [0.0, 0.0, 1.0]])
                        curr_H = curr_H @ H_step
                        if inliers is not None:
                            inlier_mask = inliers.ravel() == 1
                            curr_pts = pts_curr_good[inlier_mask].reshape(-1, 1, 2)
                        else:
                            curr_pts = pts_curr_good.reshape(-1, 1, 2)
                else:
                    curr_pts = None

                transforms_to_anchor[t] = curr_H.copy()

        # Warp all frames to anchor coordinates using Lanczos4 interpolation
        stabilized_frames: list[np.ndarray] = []
        for t, frame in enumerate(frames):
            H = transforms_to_anchor[t]
            if np.allclose(H, np.eye(3)):
                stabilized_frames.append(frame.copy())
            else:
                M_2x3 = H[:2, :]
                warped = cv2.warpAffine(
                    frame,
                    M_2x3,
                    (w, h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT,
                )
                stabilized_frames.append(warped)

        return stabilized_frames
