"""Tests for the video processor — frame extraction and clip aggregation."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.video_processor import (
    aggregate_clip_summary,
    extract_frames,
    get_video_metadata,
)

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def _make_tiny_video(path, seconds=3, fps=10, w=64, h=48):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    total = int(seconds * fps)
    for i in range(total):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = (i * 5) % 256
        out.write(frame)
    out.release()


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
def test_extract_frames():
    tmp = tempfile.mkdtemp()
    vpath = os.path.join(tmp, "test_clip.mp4")
    _make_tiny_video(vpath, seconds=3, fps=10)
    meta = get_video_metadata(vpath)
    assert meta is not None
    assert meta["fps"] > 0
    frames = extract_frames(vpath, os.path.join(tmp, "frames"), frame_interval_seconds=1.0)
    assert len(frames) >= 2
    for fi in frames:
        assert fi.width == 64
        assert fi.height == 48
        assert Path(fi.frame_path).exists()


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
def test_aggregate_clip_summary():
    frame_results = [
        {"frame_path": "f1.jpg", "timestamp_seconds": 0.0,
         "detector_label": "animal", "detector_confidence": 0.9,
         "prediction_label": "pan troglodytes", "prediction_score": 0.85,
         "review_status": "pending"},
        {"frame_path": "f2.jpg", "timestamp_seconds": 1.0,
         "detector_label": "animal", "detector_confidence": 0.8,
         "prediction_label": "pan troglodytes", "prediction_score": 0.78,
         "review_status": "pending"},
        {"frame_path": "f3.jpg", "timestamp_seconds": 2.0,
         "detector_label": None, "detector_confidence": None,
         "prediction_label": "blank", "prediction_score": 0.95,
         "review_status": "pending"},
    ]
    s = aggregate_clip_summary("clip.mp4", frame_results)
    assert s.total_frames_analyzed == 3
    assert s.animal_frame_count == 2
    assert s.blank_frame_count == 1
    assert s.human_frame_count == 0
    assert s.best_species_prediction == "pan troglodytes"
    assert s.best_species_score == 0.85
    assert s.first_detection_time == 0.0
    assert s.last_detection_time == 2.0
    assert len(s.timeline) == 3


def test_aggregate_empty():
    s = aggregate_clip_summary("empty.mp4", [])
    assert s.total_frames_analyzed == 0
    assert s.best_species_prediction is None


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
def test_metadata_invalid_video():
    assert get_video_metadata("nonexistent.mp4") is None
