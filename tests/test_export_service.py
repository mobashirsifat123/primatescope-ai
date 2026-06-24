"""Tests for the export service."""
import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import get_connection, init_db
from database.models import (
    Detection, ExportRecord, InferenceRun, MediaFile, ReviewItem, SpeciesPrediction,
)
from database.repositories import (
    DetectionRepo, ExportRepo, InferenceRepo, MediaRepo, ProjectRepo,
    PredictionRepo, ReviewRepo,
)
from services.export_service import CSV_COLUMNS, build_export_dataframe, export_csv
from utils.validation import iso_now


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    return get_connection(path), path


def _seed(conn):
    p = ProjectRepo.create(conn, "P", "d", "CHN")
    m = MediaFile(id="m1", project_id=p.id, original_filename="CS-1_test.jpg",
                  stored_path="/tmp/x.jpg", media_type="image",
                  file_size_bytes=100, station_id="CS-1",
                  uploaded_at=iso_now(), processing_status="processed")
    MediaRepo.insert(conn, m)
    run = InferenceRun(id="r1", project_id=p.id, engine="speciesnet",
                       country_code="CHN", status="success",
                       started_at=iso_now(), input_path="/tmp")
    InferenceRepo.insert(conn, run)
    d = Detection(id="d1", project_id=p.id, media_id="m1", inference_run_id="r1",
                  detector_label="animal", detector_confidence=0.9,
                  bbox_x=0.1, bbox_y=0.2, bbox_w=0.3, bbox_h=0.4,
                  bbox_format="normalized_xywh", created_at=iso_now())
    DetectionRepo.insert(conn, d)
    pr = SpeciesPrediction(id="p1", project_id=p.id, media_id="m1",
                           detection_id="d1", inference_run_id="r1",
                           prediction_label="pan troglodytes",
                           prediction_score=0.88, model_version="v4",
                           created_at=iso_now())
    PredictionRepo.insert(conn, pr)
    now = iso_now()
    ReviewRepo.upsert(conn, ReviewItem(
        id="rv1", project_id=p.id, media_id="m1", detection_id="d1",
        prediction_id="p1", queue_reason="model_prediction_review",
        review_status="approved", reviewer="Dr. A",
        reviewed_at=now, created_at=now, updated_at=now,
        final_label="animal", final_species="pan troglodytes",
    ))
    return p


def test_export_csv_columns():
    conn, path = _fresh_db()
    try:
        p = _seed(conn)
        out = tempfile.mktemp(suffix=".csv")
        count, outpath = export_csv(conn, p.id, out, reviewed_only=False)
        assert count == 1
        with open(outpath) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_COLUMNS
            row = next(reader)
            assert row[0] == p.id
            assert "pan troglodytes" in row
    finally:
        conn.close()
        os.unlink(path)


def test_export_reviewed_only():
    conn, path = _fresh_db()
    try:
        p = _seed(conn)
        out = tempfile.mktemp(suffix=".csv")
        count, _ = export_csv(conn, p.id, out, reviewed_only=True)
        assert count == 1
        exports = ExportRepo.list_by_project(conn, p.id)
        assert len(exports) == 1
        assert exports[0].row_count == 1
    finally:
        conn.close()
        os.unlink(path)


def test_build_dataframe():
    conn, path = _fresh_db()
    try:
        p = _seed(conn)
        rows = build_export_dataframe(conn, p.id, reviewed_only=False)
        assert len(rows) == 1
        assert rows[0]["prediction_label"] == "pan troglodytes"
        assert rows[0]["review_status"] == "approved"
    finally:
        conn.close()
        os.unlink(path)
