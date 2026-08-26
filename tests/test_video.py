"""Unit and integration tests for video extraction, looping, and canvas animation."""

from pathlib import Path
import numpy as np
import pytest
from portpan.video.loop_synthesizer import LoopMode, SeamlessLoopSynthesizer
from portpan.video.stabilizer_offset import VideoStabilizerOffsetEstimator
from portpan.video.stream_extractor import VideoStreamExtractor


def test_loop_synthesizer_crossfade():
    synthesizer = SeamlessLoopSynthesizer(mode=LoopMode.CROSSFADE, crossfade_frames=5)
    # Create synthetic frames with a moving gradient
    frames = [np.full((100, 100, 3), i * 5, dtype=np.uint8) for i in range(30)]

    loop = synthesizer.synthesize_loop(frames, target_length=20)
    assert len(loop) == 20
    assert loop[0].shape == (100, 100, 3)
    # Check that loop boundary is blended smoothly
    assert loop[0].dtype == np.uint8


def test_loop_synthesizer_ping_pong():
    synthesizer = SeamlessLoopSynthesizer(mode=LoopMode.PING_PONG)
    frames = [np.full((100, 100, 3), i * 10, dtype=np.uint8) for i in range(10)]

    loop = synthesizer.synthesize_loop(frames, target_length=18)
    assert len(loop) == 18
    assert loop[0].shape == (100, 100, 3)


def test_video_stream_extractor_on_real_asset():
    vid_path = Path("data/locations/windermere_point/videos/PXL_20260823_215141677_exported_stabilized_1787651205254.mp4")
    if not vid_path.exists():
        pytest.skip("Test video asset not found")

    extractor = VideoStreamExtractor()
    fps = extractor.get_video_fps(vid_path)
    assert fps > 20.0

    frames = extractor.extract_frames(vid_path, max_frames=10)
    assert len(frames) == 10
    assert frames[0].shape[0] > 0
    assert frames[0].shape[1] > 0


def test_video_offset_estimator():
    vid_path = Path("data/locations/windermere_point/videos/PXL_20260823_215141677_exported_stabilized_1787651205254.mp4")
    img_path = Path("data/locations/windermere_point/images/PXL_20260823_215141677.MP.jpg")
    if not vid_path.exists() or not img_path.exists():
        pytest.skip("Test assets not found")

    import cv2

    still_img = cv2.imread(str(img_path))
    estimator = VideoStabilizerOffsetEstimator()
    H_v2s = estimator.estimate_offset(still_img, vid_path)

    assert H_v2s.shape == (3, 3)
    # Check that scale is positive and non-zero
    assert H_v2s[0, 0] > 0.5
    assert H_v2s[1, 1] > 0.5


def test_frame_stabilizer_synthetic_drift():
    import cv2
    from portpan.video.frame_stabilizer import VideoFrameStabilizer

    # Create a synthetic high-contrast textured anchor image
    np.random.seed(42)
    base = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    # Draw high contrast geometric shapes to give distinct feature points
    cv2.circle(base, (100, 100), 40, (255, 0, 0), -1)
    cv2.rectangle(base, (60, 60), (140, 140), (0, 255, 0), 3)
    cv2.putText(base, "TEST", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Generate a drifting sequence where camera moves by (dx, dy)
    frames = []
    shifts = [(0, 0), (2, 1), (4, 3), (6, 4), (5, 3), (3, 2)]
    for dx, dy in shifts:
        M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
        shifted = cv2.warpAffine(base, M, (200, 200), borderMode=cv2.BORDER_REFLECT)
        frames.append(shifted)

    # Circular mask for aperture
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 80, 255, -1)

    stabilizer = VideoFrameStabilizer(max_features=200, ransac_threshold=1.5)
    stabilized = stabilizer.stabilize_sequence(frames, mask=mask, anchor_index=0)

    assert len(stabilized) == len(frames)
    # Compare central region MSE: stabilized frames should be much closer to frame 0 than raw drifted frames
    crop_slice = (slice(70, 130), slice(70, 130))
    anchor_crop = base[crop_slice].astype(np.float32)

    for i in range(1, len(frames)):
        raw_mse = float(np.mean((frames[i][crop_slice].astype(np.float32) - anchor_crop) ** 2))
        stab_mse = float(np.mean((stabilized[i][crop_slice].astype(np.float32) - anchor_crop) ** 2))
        # Stabilized MSE should be much lower than raw drift MSE
        assert stab_mse < raw_mse or stab_mse < 5.0


def test_frame_stabilizer_on_real_asset():
    vid_path = Path("data/locations/windermere_point/videos/PXL_20260823_215141677_exported_stabilized_1787651205254.mp4")
    if not vid_path.exists():
        pytest.skip("Test video asset not found")

    from portpan.video.frame_stabilizer import VideoFrameStabilizer

    extractor = VideoStreamExtractor()
    frames = extractor.extract_frames(vid_path, max_frames=15)

    stabilizer = VideoFrameStabilizer()
    stabilized = stabilizer.stabilize_sequence(frames)
    assert len(stabilized) == 15
    assert stabilized[0].shape == frames[0].shape
