"""PrimateScope AI — review service.

Thin wrapper around ReviewRepo.apply_action() providing named functions for
each review action. Keeps the UI and pipeline callers clean and ensures
consistent action/status mappings. Never overwrites the original AI prediction.
"""
from __future__ import annotations

from typing import Optional

from database.repositories import ReviewRepo
from utils.logging_config import get_logger

_log = get_logger("review_service")


def approve_prediction(conn, review_item_id: str, reviewer: str = "",
                       notes: Optional[str] = None):
    """Approve the AI prediction — marks review_status = 'approved'."""
    return ReviewRepo.apply_action(
        conn, review_item_id, "approve", "approved",
        reviewer=reviewer, notes=notes,
    )


def correct_prediction(conn, review_item_id: str, final_label: str,
                        final_species: Optional[str] = None,
                        reviewer: str = "", notes: Optional[str] = None):
    """Correct the AI prediction with a human-provided label."""
    return ReviewRepo.apply_action(
        conn, review_item_id, "correct", "corrected",
        reviewer=reviewer, final_label=final_label,
        final_species=final_species, notes=notes,
    )


def mark_blank(conn, review_item_id: str, reviewer: str = "",
               notes: Optional[str] = None):
    """Confirm this media as blank / no animal."""
    return ReviewRepo.apply_action(
        conn, review_item_id, "mark_blank", "blank_confirmed",
        reviewer=reviewer, final_label="blank", notes=notes,
    )


def mark_uncertain(conn, review_item_id: str, reviewer: str = "",
                   notes: Optional[str] = None):
    """Mark prediction as uncertain — needs further review or expert input."""
    return ReviewRepo.apply_action(
        conn, review_item_id, "mark_uncertain", "uncertain",
        reviewer=reviewer, notes=notes,
    )


def flag_human(conn, review_item_id: str, reviewer: str = "",
               notes: Optional[str] = None):
    """Flag this media as containing a human/person detection."""
    return ReviewRepo.apply_action(
        conn, review_item_id, "flag_human", "human_confirmed",
        reviewer=reviewer, final_label="human", notes=notes,
    )


def flag_vehicle(conn, review_item_id: str, reviewer: str = "",
                 notes: Optional[str] = None):
    """Flag this media as containing a vehicle detection."""
    return ReviewRepo.apply_action(
        conn, review_item_id, "flag_vehicle", "vehicle_confirmed",
        reviewer=reviewer, final_label="vehicle", notes=notes,
    )


def reject_prediction(conn, review_item_id: str, reviewer: str = "",
                      notes: Optional[str] = None):
    """Reject the AI prediction entirely."""
    return ReviewRepo.apply_action(
        conn, review_item_id, "reject", "rejected",
        reviewer=reviewer, notes=notes,
    )
