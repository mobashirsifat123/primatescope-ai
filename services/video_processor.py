"""PrimateScope AI — video frame extraction and clip-level aggregation.

Supports 20-30 second MP4/MOV/AVI/MKV clips. Extracts ~1 frame per second by
default (configurable), preserves the original video, and aggregates
frame-level SpeciesNet predictions into a clip summary. Never claims behavior
recognition — that is a future module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger
from utils.validation import iso_now

_log = get_logger("video_processor")


@dataclass
class FrameInfo:
    video_path: str
    frame_path: str
    frame_index: int
    timestamp_seconds: float
    extraction_time: str
    width: int
    height: int
    fps: float
    duration: float


@dataclass
class ClipSummary:
    video_file: str
    best_species_prediction: Optional[str] = None
    best_species_score: Optional[float] = None
    animal_frame_count: int = 0
    human_frame_count: int = 0
    vehicle_frame_count: int = 0
    blank_frame_count: int = 0
    total_frames_analyzed: int = 0
    best_frame_path: Optional[str] = None
    first_detection_time: Optional[float] = None
    last_detection_time: Optional[float] = None
    timeline: list[dict] = field(default_factory=list)
    review_status: str = "pending"
    notes: Optional[str] = None


def get_video_metadata(path: str | Path) -> Optional[dict]:
    """Return width/height/fps/duration for a video, or None if unreadable."""
    try:
        import cv2
    except Exception:
        _log.error("OpenCV not available for video metadata")
        return None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0
        cap.release()
        return {
            "width": width, "height": height, "fps": fps,
            "duration": duration, "frame_count": int(frame_count),
        }
    except Exception as e:
        _log.error("Video metadata error for %s: %s", path, e)
        return None


def extract_frames(
    video_path: str | Path,
    dest_dir: str | Path,
    frame_interval_seconds: float = 1.0,
    max_frames: int = 60,
) -> list[FrameInfo]:
    """Extract frames from *video_path* into *dest_dir* at the given interval.

    Defaults to 1 frame/second, capped at max_frames. Preserves the original
    video. Returns FrameInfo records for each extracted frame.
    """
    try:
        import cv2
    except Exception:
        _log.error("OpenCV not available for frame extraction")
        return []

    video_path = Path(video_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _log.error("Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0.0

    if fps <= 0:
        cap.release()
        _log.error("Invalid fps for %s", video_path)
        return []

    step = max(1, int(round(fps * frame_interval_seconds)))
    frames: list[FrameInfo] = []
    idx = 0
    extracted = 0
    now = iso_now()
    stem = video_path.stem

    while extracted < max_frames:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ret2, frame = cap.retrieve()
            if not ret2 or frame is None:
                idx += 1
                continue
            ts = idx / fps if fps else 0.0
            fname = f"{stem}_f{extracted:04d}_t{ts:06.2f}.jpg"
            fpath = dest_dir / fname
            ok = cv2.imwrite(str(fpath), frame)
            if ok:
                frames.append(FrameInfo(
                    video_path=str(video_path), frame_path=str(fpath),
                    frame_index=idx, timestamp_seconds=round(ts, 2),
                    extraction_time=now, width=width, height=height,
                    fps=round(fps, 2), duration=round(duration, 2),
                ))
                extracted += 1
        idx += 1

    cap.release()
    _log.info(
        "Extracted %d frames from %s (fps=%.2f, dur=%.1fs)",
        len(frames), video_path.name, fps, duration,
    )
    return frames


def aggregate_clip_summary(
    video_file: str,
    frame_results: list[dict],
) -> ClipSummary:
    """Aggregate frame-level parsed predictions into a clip summary.

    Each item in *frame_results* is a dict with keys: frame_path,
    timestamp_seconds, detector_label, detector_confidence, prediction_label,
    prediction_score, review_status.
    """
    summary = ClipSummary(video_file=video_file)
    summary.total_frames_analyzed = len(frame_results)
    best_score = -1.0
    first_t: Optional[float] = None
    last_t: Optional[float] = None

    for fr in frame_results:
        label = (fr.get("detector_label") or "").lower()
        pred = fr.get("prediction_label") or ""
        score = fr.get("prediction_score")
        ts = fr.get("timestamp_seconds")
        if label == "animal":
            summary.animal_frame_count += 1
        elif label == "human":
            summary.human_frame_count += 1
        elif label == "vehicle":
            summary.vehicle_frame_count += 1
        if pred and pred.lower() in ("blank", "empty"):
            summary.blank_frame_count += 1
        if score is not None and isinstance(score, (int, float)):
            if score > best_score and pred and pred.lower() not in ("blank", "empty"):
                best_score = float(score)
                summary.best_species_prediction = pred
                summary.best_species_score = round(best_score, 4)
                summary.best_frame_path = fr.get("frame_path")
        if ts is not None:
            t = float(ts)
            if first_t is None or t < first_t:
                first_t = t
            if last_t is None or t > last_t:
                last_t = t
        summary.timeline.append(fr)

    summary.first_detection_time = first_t
    summary.last_detection_time = last_t
    return summary
