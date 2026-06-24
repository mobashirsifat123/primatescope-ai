"""PrimateScope AI — repository layer for all database CRUD operations."""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Optional

from utils.logging_config import get_logger
from utils.validation import iso_now

from .models import (
    Detection,
    ExportRecord,
    InferenceRun,
    MediaFile,
    Project,
    ReviewAction,
    ReviewItem,
    SpeciesPrediction,
)

_log = get_logger("repositories")


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _row_to(conn: sqlite3.Connection, cls, table: str, row_id: str) -> Optional[Any]:
    cur = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    data = {c: row[c] for c in cols}
    try:
        return cls(**data)
    except TypeError:
        return data


class ProjectRepo:
    table = "projects"

    @staticmethod
    def create(conn, name, description="", country_code=None, study_site=None, project_id=None):
        pid = project_id or _uid("proj_")
        now = iso_now()
        conn.execute(
            "INSERT INTO projects (id,name,description,country_code,study_site,created_at,"
            "updated_at,status) VALUES (?,?,?,?,?,?,?,?)",
            (pid, name, description, country_code, study_site, now, now, "active"),
        )
        conn.commit()
        _log.info("Created project %s (%s)", pid, name)
        return Project(id=pid, name=name, description=description,
                       country_code=country_code, study_site=study_site,
                       created_at=now, updated_at=now)

    @staticmethod
    def get(conn, project_id):
        return _row_to(conn, Project, ProjectRepo.table, project_id)

    @staticmethod
    def list_all(conn):
        cur = conn.execute("SELECT * FROM projects ORDER BY created_at DESC")
        return [Project(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]

    @staticmethod
    def get_or_create_default(conn):
        rows = ProjectRepo.list_all(conn)
        if rows:
            return rows[0]
        from utils.constants import DEFAULT_COUNTRY_CODE, DEFAULT_PROJECT_NAME
        return ProjectRepo.create(conn, DEFAULT_PROJECT_NAME,
                                  "Default local project", DEFAULT_COUNTRY_CODE,
                                  study_site=None)

    @staticmethod
    def update(conn, project_id, **fields):
        if not fields:
            return
        fields["updated_at"] = iso_now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [project_id]
        conn.execute(f"UPDATE projects SET {set_clause} WHERE id = ?", values)
        conn.commit()


class MediaRepo:
    table = "media_files"

    @staticmethod
    def insert(conn, m: MediaFile):
        conn.execute(
            "INSERT INTO media_files (id,project_id,original_filename,stored_path,"
            "media_type,mime_type,file_size_bytes,width,height,duration_seconds,fps,"
            "station_id,captured_at,uploaded_at,processing_status,checksum) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (m.id, m.project_id, m.original_filename, m.stored_path, m.media_type,
             m.mime_type, m.file_size_bytes, m.width, m.height, m.duration_seconds,
             m.fps, m.station_id, m.captured_at, m.uploaded_at, m.processing_status,
             m.checksum),
        )
        conn.commit()
        return m

    @staticmethod
    def get(conn, media_id):
        return _row_to(conn, MediaFile, MediaRepo.table, media_id)

    @staticmethod
    def list_by_project(conn, project_id, media_type=None):
        if media_type:
            cur = conn.execute(
                "SELECT * FROM media_files WHERE project_id = ? AND media_type = ? "
                "ORDER BY uploaded_at DESC", (project_id, media_type))
        else:
            cur = conn.execute(
                "SELECT * FROM media_files WHERE project_id = ? "
                "ORDER BY uploaded_at DESC", (project_id,))
        return [MediaFile(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]

    @staticmethod
    def update_status(conn, media_id, status):
        conn.execute("UPDATE media_files SET processing_status = ? WHERE id = ?",
                     (status, media_id))
        conn.commit()

    @staticmethod
    def get_by_stored_path(conn, path):
        cur = conn.execute("SELECT * FROM media_files WHERE stored_path = ?", (path,))
        row = cur.fetchone()
        return MediaFile(**{k: row[k] for k in row.keys()}) if row else None

    @staticmethod
    def find_by_filename(conn, project_id, filename):
        cur = conn.execute(
            "SELECT * FROM media_files WHERE project_id = ? AND original_filename = ? "
            "ORDER BY uploaded_at DESC LIMIT 1", (project_id, filename))
        row = cur.fetchone()
        return MediaFile(**{k: row[k] for k in row.keys()}) if row else None


class InferenceRepo:
    table = "inference_runs"

    @staticmethod
    def insert(conn, run: InferenceRun):
        conn.execute(
            "INSERT INTO inference_runs (id,project_id,engine,engine_version,"
            "country_code,input_path,output_json_path,status,started_at,finished_at,"
            "duration_seconds,stdout,stderr,error_message) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run.id, run.project_id, run.engine, run.engine_version, run.country_code,
             run.input_path, run.output_json_path, run.status, run.started_at,
             run.finished_at, run.duration_seconds, run.stdout, run.stderr,
             run.error_message),
        )
        conn.commit()
        return run

    @staticmethod
    def update(conn, run_id, **fields):
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [run_id]
        conn.execute(f"UPDATE inference_runs SET {set_clause} WHERE id = ?", values)
        conn.commit()

    @staticmethod
    def get(conn, run_id):
        return _row_to(conn, InferenceRun, InferenceRepo.table, run_id)

    @staticmethod
    def list_by_project(conn, project_id, limit=50):
        cur = conn.execute(
            "SELECT * FROM inference_runs WHERE project_id = ? "
            "ORDER BY started_at DESC LIMIT ?", (project_id, limit))
        return [InferenceRun(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]


class DetectionRepo:
    table = "detections"

    @staticmethod
    def insert(conn, d: Detection):
        conn.execute(
            "INSERT INTO detections (id,project_id,media_id,inference_run_id,frame_id,"
            "detector_label,detector_confidence,bbox_x,bbox_y,bbox_w,bbox_h,bbox_format,"
            "source,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d.id, d.project_id, d.media_id, d.inference_run_id, d.frame_id,
             d.detector_label, d.detector_confidence, d.bbox_x, d.bbox_y, d.bbox_w,
             d.bbox_h, d.bbox_format, d.source, d.created_at),
        )
        return d

    @staticmethod
    def list_by_media(conn, media_id):
        cur = conn.execute(
            "SELECT * FROM detections WHERE media_id = ? "
            "ORDER BY detector_confidence DESC", (media_id,))
        return [Detection(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]


class PredictionRepo:
    table = "species_predictions"

    @staticmethod
    def insert(conn, p: SpeciesPrediction):
        conn.execute(
            "INSERT INTO species_predictions (id,project_id,media_id,detection_id,"
            "inference_run_id,prediction_label,prediction_common_name,"
            "prediction_scientific_name,prediction_score,taxonomy_level,model_version,"
            "raw_prediction_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.id, p.project_id, p.media_id, p.detection_id, p.inference_run_id,
             p.prediction_label, p.prediction_common_name,
             p.prediction_scientific_name, p.prediction_score, p.taxonomy_level,
             p.model_version, p.raw_prediction_json, p.created_at),
        )
        return p

    @staticmethod
    def list_by_media(conn, media_id):
        cur = conn.execute(
            "SELECT * FROM species_predictions WHERE media_id = ? "
            "ORDER BY prediction_score DESC", (media_id,))
        return [SpeciesPrediction(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]


class ReviewRepo:
    table = "review_items"

    @staticmethod
    def upsert(conn, item: ReviewItem):
        conn.execute(
            "INSERT INTO review_items (id,project_id,media_id,detection_id,"
            "prediction_id,queue_reason,review_status,final_label,final_species,"
            "reviewer,reviewed_at,notes,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET review_status=excluded.review_status,"
            "final_label=excluded.final_label,final_species=excluded.final_species,"
            "reviewer=excluded.reviewer,reviewed_at=excluded.reviewed_at,"
            "notes=excluded.notes,updated_at=excluded.updated_at",
            (item.id, item.project_id, item.media_id, item.detection_id,
             item.prediction_id, item.queue_reason, item.review_status,
             item.final_label, item.final_species, item.reviewer, item.reviewed_at,
             item.notes, item.created_at, item.updated_at),
        )
        conn.commit()
        return item

    @staticmethod
    def get(conn, item_id):
        return _row_to(conn, ReviewItem, ReviewRepo.table, item_id)

    @staticmethod
    def list_by_project(conn, project_id, status=None, reason=None, limit=500):
        q = "SELECT * FROM review_items WHERE project_id = ?"
        params = [project_id]
        if status:
            q += " AND review_status = ?"
            params.append(status)
        if reason:
            q += " AND queue_reason = ?"
            params.append(reason)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cur = conn.execute(q, params)
        return [ReviewItem(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]

    @staticmethod
    def list_by_media(conn, media_id):
        cur = conn.execute(
            "SELECT * FROM review_items WHERE media_id = ? "
            "ORDER BY created_at DESC", (media_id,))
        return [ReviewItem(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]

    @staticmethod
    def apply_action(conn, item_id, action, new_status, reviewer=None,
                     final_label=None, final_species=None, notes=None):
        item = ReviewRepo.get(conn, item_id)
        if item is None:
            raise ValueError(f"Review item {item_id} not found")
        old_status = item.review_status
        old_label = item.final_label
        now = iso_now()
        item.review_status = new_status
        item.reviewer = reviewer or item.reviewer
        item.reviewed_at = now
        item.updated_at = now
        if final_label is not None:
            item.final_label = final_label
        if final_species is not None:
            item.final_species = final_species
        if notes is not None:
            item.notes = notes
        conn.execute(
            "UPDATE review_items SET review_status=?,final_label=?,final_species=?,"
            "reviewer=?,reviewed_at=?,notes=?,updated_at=? WHERE id=?",
            (item.review_status, item.final_label, item.final_species, item.reviewer,
             item.reviewed_at, item.notes, item.updated_at, item.id),
        )
        act = ReviewAction(id=_uid("act_"), review_item_id=item_id, action=action,
                           old_status=old_status, new_status=new_status,
                           old_label=old_label, new_label=item.final_label,
                           reviewer=reviewer, notes=notes, created_at=now)
        conn.execute(
            "INSERT INTO review_actions (id,review_item_id,action,old_status,new_status,"
            "old_label,new_label,reviewer,notes,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (act.id, act.review_item_id, act.action, act.old_status, act.new_status,
             act.old_label, act.new_label, act.reviewer, act.notes, act.created_at),
        )
        conn.commit()
        _log.info("Review action %s on %s -> %s", action, item_id, new_status)
        return item, act

    @staticmethod
    def list_actions(conn, item_id):
        cur = conn.execute(
            "SELECT * FROM review_actions WHERE review_item_id = ? "
            "ORDER BY created_at ASC", (item_id,))
        return [ReviewAction(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]


class ExportRepo:
    table = "exports"

    @staticmethod
    def insert(conn, rec: ExportRecord):
        conn.execute(
            "INSERT INTO exports (id,project_id,export_type,export_path,row_count,"
            "created_at,filters_json) VALUES (?,?,?,?,?,?,?)",
            (rec.id, rec.project_id, rec.export_type, rec.export_path, rec.row_count,
             rec.created_at, rec.filters_json),
        )
        conn.commit()
        return rec

    @staticmethod
    def list_by_project(conn, project_id):
        cur = conn.execute(
            "SELECT * FROM exports WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,))
        return [ExportRecord(**{k: r[k] for k in r.keys()}) for r in cur.fetchall()]


class StatsRepo:
    @staticmethod
    def project_summary(conn, project_id):
        def _count(sql, params=()):
            cur = conn.execute(sql, params)
            row = cur.fetchone()
            return int(row[0]) if row else 0

        media_total = _count(
            "SELECT COUNT(*) FROM media_files WHERE project_id = ?", (project_id,))
        media_processed = _count(
            "SELECT COUNT(*) FROM media_files WHERE project_id = ? "
            "AND processing_status = 'processed'", (project_id,))
        media_images = _count(
            "SELECT COUNT(*) FROM media_files WHERE project_id = ? "
            "AND media_type = 'image'", (project_id,))
        media_videos = _count(
            "SELECT COUNT(*) FROM media_files WHERE project_id = ? "
            "AND media_type = 'video'", (project_id,))
        det_animal = _count(
            "SELECT COUNT(DISTINCT media_id) FROM detections WHERE project_id = ? "
            "AND detector_label = 'animal'", (project_id,))
        det_human = _count(
            "SELECT COUNT(DISTINCT media_id) FROM detections WHERE project_id = ? "
            "AND detector_label = 'human'", (project_id,))
        det_vehicle = _count(
            "SELECT COUNT(DISTINCT media_id) FROM detections WHERE project_id = ? "
            "AND detector_label = 'vehicle'", (project_id,))
        blank = _count(
            "SELECT COUNT(*) FROM species_predictions WHERE project_id = ? "
            "AND prediction_label = 'blank'", (project_id,))
        review_pending = _count(
            "SELECT COUNT(*) FROM review_items WHERE project_id = ? "
            "AND review_status = 'pending'", (project_id,))
        review_done = _count(
            "SELECT COUNT(*) FROM review_items WHERE project_id = ? "
            "AND review_status != 'pending'", (project_id,))
        avg_conf_row = conn.execute(
            "SELECT AVG(prediction_score) FROM species_predictions WHERE project_id = ?",
            (project_id,)).fetchone()
        avg_conf = float(avg_conf_row[0]) if avg_conf_row and avg_conf_row[0] else 0.0

        cur = conn.execute(
            "SELECT prediction_label, COUNT(*) as c FROM species_predictions "
            "WHERE project_id = ? AND prediction_label IS NOT NULL "
            "GROUP BY prediction_label ORDER BY c DESC LIMIT 10", (project_id,))
        top_species = [(r["prediction_label"], int(r["c"])) for r in cur.fetchall()]
        return {
            "media_total": media_total,
            "media_processed": media_processed,
            "media_images": media_images,
            "media_videos": media_videos,
            "detections_animal": det_animal,
            "detections_human": det_human,
            "detections_vehicle": det_vehicle,
            "blanks": blank,
            "review_pending": review_pending,
            "review_done": review_done,
            "avg_confidence": avg_conf,
            "top_species": top_species,
        }
