"""PrimateScope AI — review queue logic.

Given parsed predictions for a media file, decide the queue_reason and
review_status. Centralizes the rules so the UI and pipeline stay consistent.
"""
from __future__ import annotations

from typing import Optional

from services.result_parser import ParsedPrediction
from utils.constants import (
    QR_BLANK,
    QR_BORDERLINE,
    QR_HUMAN,
    QR_MANUAL,
    QR_MISSING,
    QR_MODEL_REVIEW,
    QR_MULTIPLE,
    QR_PARSING,
    QR_VEHICLE,
    REV_PENDING,
)


def decide_queue_reason(pred: ParsedPrediction, conf_thresh: float = 0.4) -> str:
    """Determine the queue reason for a media file from its parsed prediction."""
    if pred.has_error or pred.failures:
        return QR_PARSING
    labels = {d.detector_label for d in pred.detections if d.detector_label}
    if "human" in labels:
        return QR_HUMAN
    if "vehicle" in labels:
        return QR_VEHICLE
    pl = (pred.prediction_label or "").lower()
    if pl in ("blank", "empty"):
        return QR_BLANK
    # Count distinct non-blank species predictions across detections.
    species_set = {
        d.detector_label for d in pred.detections
        if d.detector_label and d.detector_label not in ("human", "vehicle")
    }
    if len(species_set) > 1:
        return QR_MULTIPLE
    if pred.prediction_score is None:
        return QR_MISSING
    if pred.prediction_score < conf_thresh:
        return QR_BORDERLINE
    return QR_MODEL_REVIEW
