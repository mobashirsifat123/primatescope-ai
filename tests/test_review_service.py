"""Tests for the review service module."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import init_db, get_connection
from database.models import MediaFile, ReviewItem, SpeciesPrediction, Detection
from database.repositories import (
    DetectionRepo, MediaRepo, PredictionRepo, ProjectRepo, ReviewRepo,
)
from services import review_service
from utils.validation import iso_now


@pytest.fixture
def conn():
    """In-memory SQLite database for testing."""
    import sqlite3
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    # Create tables directly from DDL.
    from database.models import SCHEMA_SQL
    for ddl in SCHEMA_SQL:
        c.execute(ddl)
    c.commit()
    yield c
    c.close()


@pytest.fixture
def setup_data(conn):
    """Create a project, media, prediction, detection, and review item."""
    now = iso_now()
    proj = ProjectRepo.create(conn, "Test Project", "desc", "USA")
    media = MediaFile(
        id="m_001", project_id=proj.id, original_filename="cam01.jpg",
        stored_path="/tmp/cam01.jpg", media_type="image",
        file_size_bytes=1024, uploaded_at=now, processing_status="needs_review",
    )
    MediaRepo.insert(conn, media)
    # Insert a minimal inference run to satisfy NOT NULL FK.
    conn.execute(
        "INSERT INTO inference_runs (id, project_id, engine, input_path, status, started_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("run_001", proj.id, "speciesnet", "/tmp/input", "success", now),
    )
    conn.commit()
    det = Detection(
        id="det_001", project_id=proj.id, media_id="m_001",
        inference_run_id="run_001", detector_label="animal",
        detector_confidence=0.85, bbox_x=0.1, bbox_y=0.2,
        bbox_w=0.3, bbox_h=0.4, source="speciesnet", created_at=now,
    )
    DetectionRepo.insert(conn, det)
    pred = SpeciesPrediction(
        id="pred_001", project_id=proj.id, media_id="m_001",
        detection_id="det_001", inference_run_id="run_001",
        prediction_label="pan troglodytes",
        prediction_score=0.88, model_version="speciesnet_v4",
        created_at=now,
    )
    PredictionRepo.insert(conn, pred)
    item = ReviewItem(
        id="rev_001", project_id=proj.id, media_id="m_001",
        detection_id="det_001", prediction_id="pred_001",
        queue_reason="model_prediction_review", review_status="pending",
        created_at=now, updated_at=now,
    )
    ReviewRepo.upsert(conn, item)
    return {"project": proj, "media": media, "detection": det,
            "prediction": pred, "review_item": item}


class TestApprove:
    def test_approve_sets_status(self, conn, setup_data):
        review_service.approve_prediction(conn, "rev_001", reviewer="Dr. A")
        item = ReviewRepo.get(conn, "rev_001")
        assert item.review_status == "approved"
        assert item.reviewer == "Dr. A"

    def test_approve_creates_audit(self, conn, setup_data):
        review_service.approve_prediction(conn, "rev_001", reviewer="Dr. A",
                                          notes="Looks correct")
        actions = ReviewRepo.list_actions(conn, "rev_001")
        assert len(actions) >= 1
        assert actions[-1].action == "approve"
        assert actions[-1].new_status == "approved"


class TestCorrect:
    def test_correct_sets_final_label(self, conn, setup_data):
        review_service.correct_prediction(
            conn, "rev_001", final_label="gorilla gorilla",
            reviewer="Dr. B", notes="Misidentified"
        )
        item = ReviewRepo.get(conn, "rev_001")
        assert item.review_status == "corrected"
        assert item.final_label == "gorilla gorilla"

    def test_original_prediction_preserved(self, conn, setup_data):
        review_service.correct_prediction(
            conn, "rev_001", final_label="gorilla gorilla", reviewer="Dr. B"
        )
        # The original AI prediction should be unchanged.
        preds = PredictionRepo.list_by_media(conn, "m_001")
        assert preds[0].prediction_label == "pan troglodytes"
        assert preds[0].prediction_score == 0.88


class TestMarkBlank:
    def test_mark_blank(self, conn, setup_data):
        review_service.mark_blank(conn, "rev_001", reviewer="Dr. C")
        item = ReviewRepo.get(conn, "rev_001")
        assert item.review_status == "blank_confirmed"
        assert item.final_label == "blank"


class TestMarkUncertain:
    def test_mark_uncertain(self, conn, setup_data):
        review_service.mark_uncertain(conn, "rev_001", reviewer="Dr. D",
                                      notes="Need expert")
        item = ReviewRepo.get(conn, "rev_001")
        assert item.review_status == "uncertain"


class TestFlagHuman:
    def test_flag_human(self, conn, setup_data):
        review_service.flag_human(conn, "rev_001", reviewer="Dr. E")
        item = ReviewRepo.get(conn, "rev_001")
        assert item.review_status == "human_confirmed"
        assert item.final_label == "human"


class TestFlagVehicle:
    def test_flag_vehicle(self, conn, setup_data):
        review_service.flag_vehicle(conn, "rev_001", reviewer="Dr. F")
        item = ReviewRepo.get(conn, "rev_001")
        assert item.review_status == "vehicle_confirmed"
        assert item.final_label == "vehicle"


class TestReject:
    def test_reject(self, conn, setup_data):
        review_service.reject_prediction(conn, "rev_001", reviewer="Dr. G",
                                         notes="Bad image quality")
        item = ReviewRepo.get(conn, "rev_001")
        assert item.review_status == "rejected"


class TestAuditTrail:
    def test_multiple_actions_create_trail(self, conn, setup_data):
        review_service.mark_uncertain(conn, "rev_001", reviewer="Dr. H")
        review_service.correct_prediction(
            conn, "rev_001", final_label="macaca mulatta", reviewer="Dr. I"
        )
        review_service.approve_prediction(conn, "rev_001", reviewer="Dr. J")
        actions = ReviewRepo.list_actions(conn, "rev_001")
        assert len(actions) >= 3
        statuses = [a.new_status for a in actions]
        assert "uncertain" in statuses
        assert "corrected" in statuses
        assert "approved" in statuses
