"""PrimateScope AI — database models and schema DDL.

Uses plain dataclasses for row representation and a single DDL list for schema
creation. Keeps the layer dependency-free (no ORM) so it is trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# DDL — executed in order by db.init_db()
# ---------------------------------------------------------------------------
SCHEMA_SQL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS projects (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        description  TEXT,
        country_code TEXT,
        study_site   TEXT,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        status       TEXT NOT NULL DEFAULT 'active'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS media_files (
        id                TEXT PRIMARY KEY,
        project_id        TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        stored_path       TEXT NOT NULL,
        media_type        TEXT NOT NULL,
        mime_type         TEXT,
        file_size_bytes   INTEGER NOT NULL DEFAULT 0,
        width             INTEGER,
        height            INTEGER,
        duration_seconds  REAL,
        fps               REAL,
        station_id        TEXT,
        captured_at       TEXT,
        uploaded_at       TEXT NOT NULL,
        processing_status TEXT NOT NULL DEFAULT 'uploaded',
        checksum          TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS inference_runs (
        id              TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL,
        engine          TEXT NOT NULL,
        engine_version  TEXT,
        country_code    TEXT,
        input_path      TEXT NOT NULL,
        output_json_path TEXT,
        status          TEXT NOT NULL,
        started_at      TEXT NOT NULL,
        finished_at     TEXT,
        duration_seconds REAL,
        stdout          TEXT,
        stderr          TEXT,
        error_message   TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detections (
        id                  TEXT PRIMARY KEY,
        project_id          TEXT NOT NULL,
        media_id            TEXT NOT NULL,
        inference_run_id    TEXT NOT NULL,
        frame_id            TEXT,
        detector_label      TEXT,
        detector_confidence REAL,
        bbox_x              REAL,
        bbox_y              REAL,
        bbox_w              REAL,
        bbox_h              REAL,
        bbox_format         TEXT,
        source              TEXT NOT NULL DEFAULT 'speciesnet',
        created_at          TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (media_id) REFERENCES media_files(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS species_predictions (
        id                       TEXT PRIMARY KEY,
        project_id               TEXT NOT NULL,
        media_id                 TEXT NOT NULL,
        detection_id             TEXT,
        inference_run_id         TEXT NOT NULL,
        prediction_label         TEXT NOT NULL,
        prediction_common_name   TEXT,
        prediction_scientific_name TEXT,
        prediction_score         REAL,
        taxonomy_level           TEXT,
        model_version            TEXT,
        raw_prediction_json      TEXT,
        created_at               TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (media_id) REFERENCES media_files(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_items (
        id            TEXT PRIMARY KEY,
        project_id    TEXT NOT NULL,
        media_id      TEXT NOT NULL,
        detection_id  TEXT,
        prediction_id TEXT,
        queue_reason  TEXT NOT NULL,
        review_status TEXT NOT NULL DEFAULT 'pending',
        final_label   TEXT,
        final_species TEXT,
        reviewer      TEXT,
        reviewed_at   TEXT,
        notes         TEXT,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (media_id) REFERENCES media_files(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_actions (
        id              TEXT PRIMARY KEY,
        review_item_id  TEXT NOT NULL,
        action          TEXT NOT NULL,
        old_status      TEXT,
        new_status      TEXT,
        old_label       TEXT,
        new_label       TEXT,
        reviewer        TEXT,
        notes           TEXT,
        created_at      TEXT NOT NULL,
        FOREIGN KEY (review_item_id) REFERENCES review_items(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exports (
        id           TEXT PRIMARY KEY,
        project_id   TEXT NOT NULL,
        export_type  TEXT NOT NULL,
        export_path  TEXT NOT NULL,
        row_count    INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL,
        filters_json TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS app_settings (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    # Indexes for common query patterns.
    "CREATE INDEX IF NOT EXISTS idx_media_project ON media_files(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_media_status ON media_files(processing_status)",
    "CREATE INDEX IF NOT EXISTS idx_runs_project ON inference_runs(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_detections_media ON detections(media_id)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_media ON species_predictions(media_id)",
    "CREATE INDEX IF NOT EXISTS idx_review_project ON review_items(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(review_status)",
    "CREATE INDEX IF NOT EXISTS idx_actions_item ON review_actions(review_item_id)",
]


# ---------------------------------------------------------------------------
# Dataclasses (row representations)
# ---------------------------------------------------------------------------
@dataclass
class Project:
    id: str
    name: str
    description: Optional[str] = None
    country_code: Optional[str] = None
    study_site: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MediaFile:
    id: str
    project_id: str
    original_filename: str
    stored_path: str
    media_type: str
    mime_type: Optional[str] = None
    file_size_bytes: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    station_id: Optional[str] = None
    captured_at: Optional[str] = None
    uploaded_at: str = ""
    processing_status: str = "uploaded"
    checksum: Optional[str] = None


@dataclass
class InferenceRun:
    id: str
    project_id: str
    engine: str
    engine_version: Optional[str] = None
    country_code: Optional[str] = None
    input_path: str = ""
    output_json_path: Optional[str] = None
    status: str = "running"
    started_at: str = ""
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class Detection:
    id: str
    project_id: str
    media_id: str
    inference_run_id: str
    frame_id: Optional[str] = None
    detector_label: Optional[str] = None
    detector_confidence: Optional[float] = None
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_w: Optional[float] = None
    bbox_h: Optional[float] = None
    bbox_format: Optional[str] = None
    source: str = "speciesnet"
    created_at: str = ""


@dataclass
class SpeciesPrediction:
    id: str
    project_id: str
    media_id: str
    inference_run_id: str
    prediction_label: str
    detection_id: Optional[str] = None
    prediction_common_name: Optional[str] = None
    prediction_scientific_name: Optional[str] = None
    prediction_score: Optional[float] = None
    taxonomy_level: Optional[str] = None
    model_version: Optional[str] = None
    raw_prediction_json: str = ""
    created_at: str = ""


@dataclass
class ReviewItem:
    id: str
    project_id: str
    media_id: str
    queue_reason: str
    review_status: str = "pending"
    detection_id: Optional[str] = None
    prediction_id: Optional[str] = None
    final_label: Optional[str] = None
    final_species: Optional[str] = None
    reviewer: Optional[str] = None
    reviewed_at: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ReviewAction:
    id: str
    review_item_id: str
    action: str
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    old_label: Optional[str] = None
    new_label: Optional[str] = None
    reviewer: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = ""


@dataclass
class ExportRecord:
    id: str
    project_id: str
    export_type: str
    export_path: str
    row_count: int = 0
    created_at: str = ""
    filters_json: Optional[str] = None
