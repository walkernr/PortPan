"""Seamless temporal loop synthesis for dynamic porthole video streams."""

from __future__ import annotations

from enum import Enum
import numpy as np


class LoopMode(str, Enum):
    """Looping synthesis mode."""

    CROSSFADE = "CROSSFADE"
    PING_PONG = "PING_PONG"


class SeamlessLoopSynthesizer:
    """Transforms raw video frame sequences into seamless, jitter-free temporal loops."""

    def __init__(self, mode: LoopMode = LoopMode.CROSSFADE, crossfade_frames: int = 15) -> None:
        self.mode = mode
        self.crossfade_frames = crossfade_frames

    def synthesize_loop(self, frames: list[np.ndarray], target_length: int = 60) -> list[np.ndarray]:
        """Generate a seamlessly looping sequence of exact length `target_length`.

        Args:
            frames: Input list of BGR video frames.
            target_length: Desired number of frames in the synthesized loop.

        Returns:
            List of seamlessly looping BGR frames of length `target_length`.
        """
        if not frames:
            raise ValueError("Input frames list is empty")

        num_input = len(frames)

        if self.mode == LoopMode.PING_PONG:
            # Forward + Backward bounce loop
            forward = frames
            backward = frames[-2:0:-1] if len(frames) > 2 else []
            full_cycle = forward + backward
            if not full_cycle:
                full_cycle = frames

            # Resample or tile to target_length
            indices = np.linspace(0, len(full_cycle) - 1, target_length).astype(np.int32)
            return [full_cycle[i] for i in indices]

        # Crossfade mode: blend the tail of the clip into the head
        xfade_len = min(self.crossfade_frames, num_input // 3, target_length // 3)
        if xfade_len < 2:
            # Clip is too short for crossfade, tile/resample directly
            indices = np.linspace(0, num_input - 1, target_length).astype(np.int32)
            return [frames[i] for i in indices]

        # Base sequence of length target_length
        raw_indices = np.linspace(0, num_input - 1, target_length + xfade_len).astype(np.int32)
        sampled_frames = [frames[i] for i in raw_indices]

        # Output buffer
        loop_frames: list[np.ndarray] = []

        for t in range(target_length):
            if t < xfade_len:
                # In the transition region: blend head (t) and tail (target_length + t)
                alpha = float(t) / float(xfade_len)
                head_f = sampled_frames[t].astype(np.float32)
                tail_f = sampled_frames[target_length + t].astype(np.float32)
                blended = (head_f * alpha + tail_f * (1.0 - alpha)).astype(np.uint8)
                loop_frames.append(blended)
            else:
                loop_frames.append(sampled_frames[t])

        return loop_frames
