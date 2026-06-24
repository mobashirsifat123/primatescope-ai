"""Tests for the database layer and repositories."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import get_connection, init_db
from database.models import MediaFile, ReviewItem
from database.repositories import (
    MediaRepo, ProjectRepo, ReviewRepo, StatsRepo, DetectionRepo, PredictionRepo,
    InferenceRepo,
)
from database.models import Detection, InferenceRun, SpeciesPrediction
from utils.validation import iso_now

import tempfile
import os


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    return get_connection(path), path


def test_safe_filename_via_storage():
    from utils.validation import safe_filename
    assert safe_filename("CS-1 test file.jpg") == "CS-1_test_file.jpg"
    assert safe_filename("normal.png") == "normal.png"
    assert safe_filename("a b/c:d?.txt") == "a_b_c_d.txt"


def test_project_creation():
    conn, path = _fresh_db()
    try:
        p = ProjectRepo.create(conn, "TestProj", "desc", "CHN")
        assert p.name == "TestProj"
        assert p.country_code == "CHN"
        fetched = ProjectRepo.get(conn, p.id)
        assert fetched is not None
        assert fetched.id == p.id
        allp = ProjectRepo.list_all(conn)
        assert len(allp) == 1
    finally:
        conn.close()
        os.unlink(path)


def test_media_insert_and_query():
    conn, path = _fresh_db()
    try:
        p = ProjectRepo.create(conn, "P", "d", "BGD")
        m = MediaFile(
            id="m1", project_id=p.id, original_filename="CS-1_test.jpg",
            stored_path="/tmp/x.jpg", media_type="image", file_size_bytes=100,
            uploaded_at=iso_now(),
        )
        MediaRepo.insert(conn, m)
        got = MediaRepo.get(conn, "m1")
        assert got is not None
        assert got.original_filename == "CS-1_test.jpg"
        listed = MediaRepo.list_by_project(conn, p.id)
        assert len(listed) == 1
        by_path = MediaRepo.get_by_stored_path(conn, "/tmp/x.jpg")
        assert by_path.id == "m1"
    finally:
        conn.close()
        os.unlink(path)


def test_review_status_update_and_audit():
    conn, path = _fresh_db()
    try:
        p = ProjectRepo.create(conn, "P", "d", "BGD")
        m = MediaFile(id="m1", project_id=p.id, original_filename="x.jpg",
                      stored_path="/tmp/x.jpg", media_type="image",
                      file_size_bytes=1, uploaded_at=iso_now())
        MediaRepo.insert(conn, m)
        now = iso_now()
        item = ReviewItem(
            id="r1", project_id=p.id, media_id="m1", queue_reason="borderline_confidence",
            review_status="pending", created_at=now, updated_at=now,
        )
        ReviewRepo.upsert(conn, item)
        item2, act = ReviewRepo.apply_action(
            conn, "r1", action="approve", new_status="approved",
            reviewer="Dr. A", notes="looks good",
        )
        assert item2.review_status == "approved"
        assert item2.reviewer == "Dr. A"
        assert item2.notes == "looks good"
        actions = ReviewRepo.list_actions(conn, "r1")
        assert len(actions) == 1
        assert actions[0].action == "approve"
        assert actions[0].old_status == "pending"
        assert actions[0].new_status == "approved"
    finally:
        conn.close()
        os.unlink(path)


def test_stats_summary():
    conn, path = _fresh_db()
    try:
        p = ProjectRepo.create(conn, "P", "d", "CHN")
        m = MediaFile(id="m1", project_id=p.id, original_filename="x.jpg",
                      stored_path="/tmp/x.jpg", media_type="image",
                      file_size_bytes=1, uploaded_at=iso_now(),
                      processing_status="processed")
        MediaRepo.insert(conn, m)
        run = InferenceRun(id="r1", project_id=p.id, engine="speciesnet",
                           status="success", started_at=iso_now(), input_path="/tmp")
        InferenceRepo.insert(conn, run)
        d = Detection(id="d1", project_id=p.id, media_id="m1",
                      inference_run_id="r1", detector_label="animal",
                      detector_confidence=0.9, created_at=iso_now())
        DetectionRepo.insert(conn, d)
        pr = SpeciesPrediction(id="p1", project_id=p.id, media_id="m1",
                               inference_run_id="r1", prediction_label="pan troglodytes",
                               prediction_score=0.88, created_at=iso_now())
        PredictionRepo.insert(conn, pr)
        s = StatsRepo.project_summary(conn, p.id)
        assert s["media_total"] == 1
        assert s["media_processed"] == 1
        assert s["detections_animal"] == 1
        assert s["avg_confidence"] == 0.88
        assert len(s["top_species"]) == 1
        assert s["top_species"][0][0] == "pan troglodytes"
    finally:
        conn.close()
        os.unlink(path)


def test_get_or_create_default():
    conn, path = _fresh_db()
    try:
        p = ProjectRepo.get_or_create_default(conn)
        assert p is not None
        p2 = ProjectRepo.get_or_create_default(conn)
        assert p2.id == p.id
    finally:
        conn.close()
        os.unlink(path)
