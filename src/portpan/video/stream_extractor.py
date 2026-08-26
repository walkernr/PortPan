"""Video stream extraction and frame standardization."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from portpan.core.exceptions import VideoProcessingError


class VideoStreamExtractor:
    """Extracts and standardizes video streams from Live Photo MP4 files."""

    def __init__(self, target_fps: float = 30.0) -> None:
        self.target_fps = target_fps

    def extract_frames(self, video_path: Path | str, max_frames: int | None = None) -> list[np.ndarray]:
        """Read all video frames from the given video file.

        Args:
            video_path: Path to the input video file.
            max_frames: Optional upper limit on the number of frames to extract.

        Returns:
            List of BGR frames as NumPy uint8 arrays.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file: {video_path}")

        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break

        cap.release()

        if not frames:
            raise VideoProcessingError(f"No frames could be extracted from: {video_path}")

        return frames

    def get_video_fps(self, video_path: Path | str) -> float:
        """Return the native frame rate of the video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps if fps > 0 else self.target_fps
