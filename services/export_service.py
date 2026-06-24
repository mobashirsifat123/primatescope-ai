"""PrimateScope AI — CSV export service for reviewed research data.

Exports detection + prediction + review records as a flat CSV suitable for
downstream statistical analysis. By default only reviewed rows are exported;
an option includes all predictions with their review_status.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Optional

from database.models import ExportRecord
from database.repositories import ExportRepo
from utils.logging_config import get_logger
from utils.validation import iso_now
import uuid

_log = get_logger("export_service")

CSV_COLUMNS = [
    "project_id",
    "media_id",
    "original_filename",
    "stored_path",
    "media_type",
    "station_id",
    "captured_at",
    "uploaded_at",
    "detector_label",
    "detector_confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "prediction_label",
    "prediction_common_name",
    "prediction_scientific_name",
    "prediction_score",
    "taxonomy_level",
    "model_version",
    "review_status",
    "final_label",
    "final_species",
    "reviewer",
    "reviewed_at",
    "notes",
    "queue_reason",
    "inference_run_id",
    "engine",
    "country_code",
    "created_at",
]

_EXPORT_QUERY = """
SELECT
    m.project_id        AS project_id,
    m.id                AS media_id,
    m.original_filename AS original_filename,
    m.stored_path       AS stored_path,
    m.media_type        AS media_type,
    m.station_id        AS station_id,
    m.captured_at       AS captured_at,
    m.uploaded_at       AS uploaded_at,
    d.detector_label    AS detector_label,
    d.detector_confidence AS detector_confidence,
    d.bbox_x            AS bbox_x,
    d.bbox_y            AS bbox_y,
    d.bbox_w            AS bbox_w,
    d.bbox_h            AS bbox_h,
    sp.prediction_label AS prediction_label,
    sp.prediction_common_name AS prediction_common_name,
    sp.prediction_scientific_name AS prediction_scientific_name,
    sp.prediction_score AS prediction_score,
    sp.taxonomy_level   AS taxonomy_level,
    sp.model_version    AS model_version,
    ri.review_status    AS review_status,
    ri.final_label      AS final_label,
    ri.final_species    AS final_species,
    ri.reviewer         AS reviewer,
    ri.reviewed_at      AS reviewed_at,
    ri.notes            AS notes,
    ri.queue_reason     AS queue_reason,
    sp.inference_run_id AS inference_run_id,
    ir.engine           AS engine,
    ir.country_code     AS country_code,
    sp.created_at       AS created_at
FROM species_predictions sp
JOIN media_files m        ON sp.media_id = m.id
LEFT JOIN detections d    ON sp.detection_id = d.id
LEFT JOIN review_items ri ON ri.media_id = sp.media_id
LEFT JOIN inference_runs ir ON sp.inference_run_id = ir.id
WHERE m.project_id = ?
"""


def export_csv(
    conn: sqlite3.Connection,
    project_id: str,
    export_path: str | Path,
    reviewed_only: bool = True,
) -> tuple[int, Path]:
    """Export predictions to CSV. Returns (row_count, path)."""
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    query = _EXPORT_QUERY
    params: list = [project_id]
    if reviewed_only:
        query += " AND ri.review_status IS NOT NULL AND ri.review_status != 'pending'"
    query += " ORDER BY m.uploaded_at ASC"

    cur = conn.execute(query, params)
    rows = cur.fetchall()
    written = 0
    with open(export_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for row in rows:
            w.writerow([row[c] if row[c] is not None else "" for c in CSV_COLUMNS])
            written += 1

    ExportRepo.insert(conn, ExportRecord(
        id=f"exp_{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        export_type="reviewed_csv" if reviewed_only else "all_predictions_csv",
        export_path=str(export_path),
        row_count=written,
        created_at=iso_now(),
        filters_json=f'{{"reviewed_only": {str(reviewed_only).lower()}}}',
    ))
    _log.info("Exported %d rows to %s", written, export_path)
    return written, export_path


def build_export_dataframe(conn: sqlite3.Connection, project_id: str,
                           reviewed_only: bool = False) -> "list[dict]":
    """Return export rows as a list of dicts (for in-app preview)."""
    query = _EXPORT_QUERY
    params: list = [project_id]
    if reviewed_only:
        query += " AND ri.review_status IS NOT NULL AND ri.review_status != 'pending'"
    query += " ORDER BY m.uploaded_at ASC LIMIT 200"
    cur = conn.execute(query, params)
    return [{c: row[c] for c in CSV_COLUMNS} for row in cur.fetchall()]
