"""PrimateScope AI — bounding-box overlay drawing for camera-trap images.

Draws detection boxes on an in-memory PIL image copy using the Obsidian Canopy
palette. Never modifies the original file. Handles both normalized and absolute
bbox formats, and missing bboxes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from utils.constants import (
    BBOX_COLOR_ANIMAL,
    BBOX_COLOR_HUMAN,
    BBOX_COLOR_LOW_CONF,
    BBOX_COLOR_VEHICLE,
    CLR_AMBER,
    CLR_BORDER,
    CLR_TEAL,
    CLR_WHITE,
    LOW_CONF_THRESHOLD,
)
from utils.logging_config import get_logger

_log = get_logger("bbox_draw")


def _box_color(label: Optional[str], conf: Optional[float]) -> str:
    lbl = (label or "").lower()
    if lbl == "human":
        return BBOX_COLOR_HUMAN
    if lbl == "vehicle":
        return BBOX_COLOR_VEHICLE
    if conf is not None and conf < LOW_CONF_THRESHOLD:
        return BBOX_COLOR_LOW_CONF
    return BBOX_COLOR_ANIMAL


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _font(size: int = 12) -> ImageFont.ImageFont:
    for name in [
        "Menlo", "Consolas", "DejaVu Sans Mono", "Courier New", "monospace"
    ]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_bboxes_on_image(
    image_path: str | Path,
    detections: list[dict],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[Image.Image]:
    """Draw bounding boxes on a copy of the image at *image_path*.

    Each detection is a dict with keys: detector_label, detector_confidence,
    bbox_x, bbox_y, bbox_w, bbox_h, bbox_format (optional, default
    normalized_xywh), prediction_label (optional).
    Returns a PIL Image, or None on failure.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        _log.error("Cannot open image %s: %s", image_path, e)
        return None
    iw, ih = img.size
    draw = ImageDraw.Draw(img)
    font = _font(max(11, min(16, iw // 80)))
    for det in detections:
        bx = det.get("bbox_x")
        by = det.get("bbox_y")
        bw = det.get("bbox_w")
        bh = det.get("bbox_h")
        if None in (bx, by, bw, bh):
            continue
        fmt = det.get("bbox_format", "normalized_xywh")
        if fmt == "normalized_xywh":
            px, py, pw, ph = bx * iw, by * ih, bw * iw, bh * ih
        else:
            px, py, pw, ph = bx, by, bw, bh
        label = det.get("detector_label")
        conf = det.get("detector_confidence")
        color = _hex_to_rgb(_box_color(label, conf))
        # Box
        draw.rectangle([px, py, px + pw, py + ph], outline=color, width=max(2, iw // 400))
        # Label
        pred = det.get("prediction_label")
        parts = []
        if label:
            parts.append(label)
        if conf is not None:
            parts.append(f"{conf:.2f}")
        if pred and pred.lower() not in (label or "").lower():
            parts.append(pred)
        text = " ".join(parts) if parts else "detection"
        tw = font.getlength(text) + 8
        th = 16
        ty = max(0, py - th)
        draw.rectangle([px, ty, px + tw, ty + th], fill=color)
        draw.text((px + 4, ty + 1), text, fill=(0, 0, 0), font=font)
    return img


def best_video_frame_thumbnail(
    frame_path: str | Path,
    detections: list[dict] | None = None,
) -> Optional[Image.Image]:
    """Open a video frame and optionally draw boxes. Returns PIL image or None."""
    if not detections:
        try:
            return Image.open(frame_path).convert("RGB")
        except Exception as e:
            _log.error("Cannot open frame %s: %s", frame_path, e)
            return None
    return draw_bboxes_on_image(frame_path, detections)
