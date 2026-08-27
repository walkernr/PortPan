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

        # Base sequence of length target_length with smooth temporal interpolation
        raw_times = np.linspace(0, num_input - 1, target_length + xfade_len)
        sampled_frames: list[np.ndarray] = []
        for t_val in raw_times:
            idx0 = int(np.floor(t_val))
            idx1 = min(idx0 + 1, num_input - 1)
            frac = float(t_val - idx0)
            if frac < 1e-4 or idx0 == idx1:
                sampled_frames.append(frames[idx0])
            else:
                f0 = frames[idx0].astype(np.float32)
                f1 = frames[idx1].astype(np.float32)
                sampled_frames.append((f0 * (1.0 - frac) + f1 * frac).astype(np.uint8))

        # Output buffer
        loop_frames: list[np.ndarray] = []

        for t in range(target_length):
            loop_frames.append(self.synthesize_frame(frames, t, target_length=target_length))

        return loop_frames

    def synthesize_frame(self, frames: list[np.ndarray], t: int, target_length: int = 60) -> np.ndarray:
        """Generate a single synthesized frame at index `t` on the fly with zero memory overhead.

        Args:
            frames: Input list of BGR video frames.
            t: Frame index in range [0, target_length - 1].
            target_length: Total frames in the looping sequence.

        Returns:
            Synthesized BGR frame at time `t`.
        """
        if not frames:
            raise ValueError("Input frames list is empty")

        num_input = len(frames)

        if self.mode == LoopMode.PING_PONG:
            forward = frames
            backward = frames[-2:0:-1] if len(frames) > 2 else []
            full_cycle = forward + backward
            if not full_cycle:
                full_cycle = frames
            raw_time = t * (len(full_cycle) - 1) / max(1, target_length - 1)
            idx0 = int(np.floor(raw_time))
            idx1 = min(idx0 + 1, len(full_cycle) - 1)
            frac = float(raw_time - idx0)
            if frac < 1e-4 or idx0 == idx1:
                return full_cycle[idx0]
            return (full_cycle[idx0].astype(np.float32) * (1.0 - frac) + full_cycle[idx1].astype(np.float32) * frac).astype(np.uint8)

        # Crossfade mode
        xfade_len = min(self.crossfade_frames, num_input // 3, target_length // 3)
        if xfade_len < 2:
            raw_time = t * (num_input - 1) / max(1, target_length - 1)
            idx0 = int(np.floor(raw_time))
            idx1 = min(idx0 + 1, num_input - 1)
            frac = float(raw_time - idx0)
            if frac < 1e-4 or idx0 == idx1:
                return frames[idx0]
            return (frames[idx0].astype(np.float32) * (1.0 - frac) + frames[idx1].astype(np.float32) * frac).astype(np.uint8)

        # Helper to sample at time slot k in [0, target_length + xfade_len - 1]
        total_slots = target_length + xfade_len - 1
        def sample_slot(k: int) -> np.ndarray:
            r_t = k * (num_input - 1) / max(1, total_slots)
            i0 = int(np.floor(r_t))
            i1 = min(i0 + 1, num_input - 1)
            fr = float(r_t - i0)
            if fr < 1e-4 or i0 == i1:
                return frames[i0]
            return (frames[i0].astype(np.float32) * (1.0 - fr) + frames[i1].astype(np.float32) * fr).astype(np.uint8)

        if t < xfade_len:
            alpha = float(t) / float(xfade_len)
            head_f = sample_slot(t).astype(np.float32)
            tail_f = sample_slot(target_length + t).astype(np.float32)
            return (head_f * alpha + tail_f * (1.0 - alpha)).astype(np.uint8)

        return sample_slot(t)
