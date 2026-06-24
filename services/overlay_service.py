"""PrimateScope AI — overlay service for camera-trap image annotation.

Wraps the bbox_draw module with Obsidian Canopy color rules and provides a
clean API: create_image_overlay(). Never modifies the original image file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from services.bbox_draw import draw_bboxes_on_image, best_video_frame_thumbnail
from utils.logging_config import get_logger

_log = get_logger("overlay_service")


def create_image_overlay(
    image_path: str | Path,
    detections: list[dict],
    save_path: Optional[str | Path] = None,
) -> Optional[Image.Image]:
    """Draw bounding boxes on a copy of the image and optionally save.

    Uses Obsidian Canopy colors:
      - Teal (#0adec8) for animal detections
      - Amber (#FFB84D) for low-confidence detections
      - Warning amber for human/person detections
      - Slate/amber for vehicle detections

    Parameters
    ----------
    image_path : path to the original camera-trap image
    detections : list of dicts with keys: detector_label, detector_confidence,
                 bbox_x, bbox_y, bbox_w, bbox_h, bbox_format, prediction_label
    save_path  : if given, save the overlay image to this path

    Returns
    -------
    PIL Image with overlays drawn, or None on failure.
    """
    img = draw_bboxes_on_image(image_path, detections)
    if img is None:
        return None

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img.save(str(save_path), quality=90)
            _log.info("Overlay saved to %s", save_path)
        except Exception as e:
            _log.error("Failed to save overlay to %s: %s", save_path, e)

    return img


def create_video_frame_overlay(
    frame_path: str | Path,
    detections: list[dict] | None = None,
) -> Optional[Image.Image]:
    """Open a video frame and optionally draw detection boxes.

    Returns a PIL Image or None.
    """
    return best_video_frame_thumbnail(frame_path, detections)
