"""PrimateScope AI — production inference pipeline orchestration.

Coordinates file storage, SpeciesNet inference, result parsing, database
persistence, and review-item creation for a batch of uploaded media.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from database.models import (
    Detection, InferenceRun, MediaFile, ReviewItem, SpeciesPrediction,
)
from database.repositories import (
    DetectionRepo, InferenceRepo, MediaRepo, PredictionRepo, ReviewRepo,
)
from services.file_storage import FileStorage
from services.queue_logic import decide_queue_reason
from services.result_parser import ParseResult, parse_speciesnet_output
from services.speciesnet_runner import InferenceRunResult, run_speciesnet_on_folder
from services.video_processor import aggregate_clip_summary, extract_frames, get_video_metadata
from utils.constants import (
    MEDIA_IMAGE, MEDIA_VIDEO, PROC_FAILED, PROC_NEEDS_REVIEW,
    PROC_PROCESSING, PROC_UPLOADED, REV_PENDING, RUN_FAILED, RUN_RUNNING, RUN_SUCCESS,
)
from utils.logging_config import get_logger
from utils.validation import get_media_type, iso_now

_log = get_logger("pipeline")


@dataclass
class BatchResult:
    project_id: str
    inference_run_id: Optional[str] = None
    media_count: int = 0
    image_count: int = 0
    video_count: int = 0
    detection_count: int = 0
    prediction_count: int = 0
    review_item_count: int = 0
    inference_success: bool = False
    inference_error: Optional[str] = None
    inference_result: Optional[InferenceRunResult] = None
    parse_result: Optional[ParseResult] = None
    video_summaries: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def save_uploaded_files(conn, storage, project_id, uploaded_files):
    """Persist Streamlit UploadedFile objects and return MediaFile records."""
    out = []
    for uf in uploaded_files:
        name = uf.name
        mtype = get_media_type(name)
        if mtype is None:
            _log.warning("Skipping unsupported file: %s", name)
            continue
        try:
            info = storage.save_stream(project_id, uf.getvalue(), name)
        except ValueError as e:
            _log.error("Save failed for %s: %s", name, e)
            continue
        mid = f"media_{uuid.uuid4().hex[:12]}"
        w = h = dur = fps = None
        if mtype == MEDIA_VIDEO:
            meta = get_video_metadata(info["stored_path"])
            if meta:
                w, h, dur, fps = meta["width"], meta["height"], meta["duration"], meta["fps"]
        elif mtype == MEDIA_IMAGE:
            try:
                from PIL import Image
                with Image.open(info["stored_path"]) as img:
                    w, h = img.size
            except Exception:
                pass
        m = MediaFile(
            id=mid, project_id=project_id, original_filename=info["original_filename"],
            stored_path=info["stored_path"], media_type=mtype,
            file_size_bytes=info["file_size_bytes"], width=w, height=h,
            duration_seconds=dur, fps=fps, station_id=info["station_id"],
            captured_at=info["captured_at"], uploaded_at=iso_now(),
            processing_status=PROC_UPLOADED, checksum=info["checksum"],
        )
        MediaRepo.insert(conn, m)
        out.append(m)
    return out


def _persist_prediction(conn, project_id, media_id, run_id, pred, det_thresh=0.25, review_thresh=0.70):
    """Persist detections + prediction + review item for one parsed image/frame."""
    now = iso_now()
    dets = []
    for d in pred.detections:
        # Filter MegaDetector boxes by detection threshold
        if d.detector_confidence is not None and d.detector_confidence < det_thresh:
            continue
        did = f"det_{uuid.uuid4().hex[:12]}"
        det = Detection(
            id=did, project_id=project_id, media_id=media_id,
            inference_run_id=run_id, detector_label=d.detector_label,
            detector_confidence=d.detector_confidence,
            bbox_x=d.bbox_x, bbox_y=d.bbox_y, bbox_w=d.bbox_w, bbox_h=d.bbox_h,
            bbox_format=d.bbox_format, source="speciesnet", created_at=now,
        )
        DetectionRepo.insert(conn, det)
        dets.append(det)
    pid = f"pred_{uuid.uuid4().hex[:12]}"
    raw = json.dumps(pred.raw, ensure_ascii=False)[:65000]
    p = SpeciesPrediction(
        id=pid, project_id=project_id, media_id=media_id,
        detection_id=dets[0].id if dets else None, inference_run_id=run_id,
        prediction_label=pred.prediction_label or "unknown",
        prediction_common_name=pred.common_name,
        prediction_scientific_name=pred.scientific_name,
        prediction_score=pred.prediction_score, taxonomy_level=pred.taxonomy_level,
        model_version=pred.model_version, raw_prediction_json=raw, created_at=now,
    )
    PredictionRepo.insert(conn, p)
    reason = decide_queue_reason(pred, review_thresh)
    rid = f"rev_{uuid.uuid4().hex[:12]}"
    item = ReviewItem(
        id=rid, project_id=project_id, media_id=media_id,
        detection_id=dets[0].id if dets else None, prediction_id=pid,
        queue_reason=reason, review_status=REV_PENDING,
        created_at=now, updated_at=now,
    )
    ReviewRepo.upsert(conn, item)
    return dets, [p], item


def _match_pred(pr, media):
    """Match a parsed prediction to a media file by path or filename."""
    tp, tn, ts = media.stored_path, media.original_filename, Path(media.original_filename).stem
    for pred in pr.predictions:
        if pred.filepath == tp or Path(pred.filepath).name == tn or Path(pred.filepath).stem == ts:
            return pred
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    nt = norm(tn)
    if nt:
        for pred in pr.predictions:
            if norm(Path(pred.filepath).name) == nt:
                return pred
    return None


def _reuse_or_run(output_json, project_id, folder, country, force, now, engine):
    if output_json.exists() and not force:
        return InferenceRunResult(True, 0, str(output_json), "(reused)", "", None, 0.0, now, now, "(reused)")
    if engine == "md_and_speciesnet":
        from services.speciesnet_runner import run_md_and_speciesnet
        return run_md_and_speciesnet(project_id, folder, output_json, country)
    return run_speciesnet_on_folder(project_id, folder, output_json, country)


def run_image_pipeline(conn, storage, project_id, image_media, country=None, force=False, det_thresh=0.25, review_thresh=0.70, engine="md_and_speciesnet"):
    result = BatchResult(project_id=project_id, image_count=len(image_media))
    if not image_media:
        result.inference_error = "No image files to process."
        return result
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    in_dir = storage.originals_dir(project_id)
    out_json = storage.outputs_dir(project_id) / f"{run_id}.json"
    now = iso_now()
    InferenceRepo.insert(conn, InferenceRun(
        id=run_id, project_id=project_id, engine=engine, country_code=country,
        input_path=str(in_dir), output_json_path=str(out_json),
        status=RUN_RUNNING, started_at=now,
    ))
    inf = _reuse_or_run(out_json, project_id, in_dir, country, force, now, engine)
    result.inference_result = inf
    result.inference_success = inf.success
    result.inference_error = inf.error_message
    if not inf.success:
        InferenceRepo.update(conn, run_id, status=RUN_FAILED, finished_at=iso_now(),
                             stderr=inf.stderr, error_message=inf.error_message,
                             duration_seconds=inf.duration_seconds)
        for m in image_media:
            MediaRepo.update_status(conn, m.id, PROC_FAILED)
        return result
    pr = parse_speciesnet_output(out_json)
    result.parse_result = pr
    dc = pc = rc = 0
    for media in image_media:
        matched = _match_pred(pr, media)
        if matched is None:
            result.errors.append(f"No prediction found for {media.original_filename}")
            n2 = iso_now()
            ReviewRepo.upsert(conn, ReviewItem(
                id=f"rev_{uuid.uuid4().hex[:12]}", project_id=project_id,
                media_id=media.id, queue_reason="missing_prediction",
                review_status=REV_PENDING, created_at=n2, updated_at=n2,
            ))
            rc += 1
            MediaRepo.update_status(conn, media.id, PROC_NEEDS_REVIEW)
            continue
        dets, preds, _ = _persist_prediction(conn, project_id, media.id, run_id, matched, conf_thresh)
        dc += len(dets); pc += len(preds); rc += 1
        MediaRepo.update_status(conn, media.id, PROC_NEEDS_REVIEW)
    InferenceRepo.update(conn, run_id, status=RUN_SUCCESS, finished_at=iso_now(),
                         duration_seconds=inf.duration_seconds, stdout=inf.stdout, stderr=inf.stderr)
    result.detection_count = dc
    result.prediction_count = pc
    result.review_item_count = rc
    return result


def run_video_pipeline(conn, storage, project_id, video_media, country=None, force=False, frame_interval=1.0, det_thresh=0.25, review_thresh=0.70, engine="md_and_speciesnet"):
    """Extract frames from each video, run SpeciesNet on frames, aggregate."""
    result = BatchResult(project_id=project_id, video_count=len(video_media))
    if not video_media:
        result.inference_error = "No video files to process."
        return result
    frames_root = storage.uploads / project_id / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    now = iso_now()
    out_json = storage.outputs_dir(project_id) / f"{run_id}.json"
    InferenceRepo.insert(conn, InferenceRun(
        id=run_id, project_id=project_id, engine=engine,
        country_code=country, input_path=str(frames_root),
        output_json_path=str(out_json), status=RUN_RUNNING, started_at=now,
    ))
    frame_to_media, frame_to_ts = {}, {}
    for vm in video_media:
        frames = extract_frames(
            vm.stored_path,
            storage.frames_dir(project_id, Path(vm.original_filename).stem),
            frame_interval_seconds=frame_interval,
        )
        for fi in frames:
            frame_to_media[str(fi.frame_path)] = vm.id
            frame_to_ts[str(fi.frame_path)] = fi.timestamp_seconds
        MediaRepo.update_status(conn, vm.id, PROC_PROCESSING)
    if not frame_to_media:
        InferenceRepo.update(conn, run_id, status=RUN_FAILED, finished_at=iso_now(),
                             error_message="No frames could be extracted.")
        result.inference_error = "No frames could be extracted from videos."
        for vm in video_media:
            MediaRepo.update_status(conn, vm.id, PROC_FAILED)
        return result
    inf = _reuse_or_run(out_json, project_id, frames_root, country, force, now, engine)
    result.inference_result = inf
    result.inference_success = inf.success
    result.inference_error = inf.error_message
    if not inf.success:
        InferenceRepo.update(conn, run_id, status=RUN_FAILED, finished_at=iso_now(),
                             stderr=inf.stderr, error_message=inf.error_message,
                             duration_seconds=inf.duration_seconds)
        for vm in video_media:
            MediaRepo.update_status(conn, vm.id, PROC_FAILED)
        return result
    pr = parse_speciesnet_output(out_json)
    result.parse_result = pr
    dc = pc = rc = 0
    per_media = {}
    for pred in pr.predictions:
        fp = pred.filepath
        mid = frame_to_media.get(fp)
        # Try to match the relative path from MD to the absolute paths in frame_to_media
        if mid is None:
            for k, v in frame_to_media.items():
                if k.endswith(fp) or Path(k).name == Path(fp).name:
                    mid = v
                    fp = k  # Use the absolute path for DB storage
                    pred.filepath = k
                    break
        
        # If still None, it might be an image run
        if mid is None:
            name = Path(fp).name
            m = MediaRepo.find_by_filename(conn, project_id, name)
            if m:
                mid = m.id
        
        if mid is None:
            result.errors.append(f"Unmatched frame: {pred.filename}")
            continue
        fd = pred.detections[0] if pred.detections else None
        per_media.setdefault(mid, []).append({
            "frame_path": fp, "timestamp_seconds": frame_to_ts.get(fp),
            "detector_label": fd.detector_label if fd else None,
            "detector_confidence": fd.detector_confidence if fd else None,
            "prediction_label": pred.prediction_label,
            "prediction_score": pred.prediction_score,
            "review_status": REV_PENDING,
        })
        _persist_prediction(conn, project_id, mid, run_id, pred, det_thresh, review_thresh)
        dc += len(pred.detections); pc += 1
    for vm in video_media:
        rows = per_media.get(vm.id, [])
        summary = aggregate_clip_summary(vm.original_filename, rows)
        result.video_summaries.append(summary)
        reason = "model_prediction_review"
        if summary.human_frame_count > 0:
            reason = "human_detected"
        elif summary.vehicle_frame_count > 0:
            reason = "vehicle_detected"
        elif summary.total_frames_analyzed > 0 and summary.blank_frame_count == summary.total_frames_analyzed:
            reason = "blank"
        n2 = iso_now()
        ReviewRepo.upsert(conn, ReviewItem(
            id=f"rev_{uuid.uuid4().hex[:12]}" , project_id=project_id, media_id=vm.id,
            queue_reason=reason, review_status=REV_PENDING,
            notes=f"Clip: best={summary.best_species_prediction} ({summary.best_species_score})",
            created_at=n2, updated_at=n2,
        ))
        rc += 1
        MediaRepo.update_status(conn, vm.id, PROC_NEEDS_REVIEW)
    InferenceRepo.update(conn, run_id, status=RUN_SUCCESS, finished_at=iso_now(),
                         duration_seconds=inf.duration_seconds, stdout=inf.stdout, stderr=inf.stderr)
    result.detection_count = dc
    result.prediction_count = pc
    result.review_item_count = rc
    return result


def run_full_batch(conn, storage, project_id, uploaded_files, country,
                   force=False, frame_interval=1.0, det_thresh=0.25, review_thresh=0.70, engine="md_and_speciesnet"):
    """End-to-end: save uploads, split images/videos, run both pipelines."""
    yield "Saving uploaded files to disk..."
    media = save_uploaded_files(conn, storage, project_id, uploaded_files)
    images = [m for m in media if m.media_type == MEDIA_IMAGE]
    videos = [m for m in media if m.media_type == MEDIA_VIDEO]
    
    ir = BatchResult(project_id=project_id)
    if images:
        yield f"Running image pipeline ({len(images)} images) using {engine}..."
        ir = run_image_pipeline(conn, storage, project_id, images, country, force, det_thresh, review_thresh, engine)
        yield "Parsed results and saved database records for images."
        
    vr = BatchResult(project_id=project_id)
    if videos:
        yield f"Extracting frames and running video pipeline ({len(videos)} videos)..."
        vr = run_video_pipeline(conn, storage, project_id, videos, country, force, frame_interval, det_thresh, review_thresh, engine)
        yield "Processed video clip frames and saved records."
        
    yield "Preparing final batch results..."
    c = BatchResult(project_id=project_id)
    c.image_count = len(images)
    c.video_count = len(videos)
    c.media_count = len(media)
    c.detection_count = ir.detection_count + vr.detection_count
    c.prediction_count = ir.prediction_count + vr.prediction_count
    c.review_item_count = ir.review_item_count + vr.review_item_count
    c.inference_success = ir.inference_success or vr.inference_success
    c.inference_error = ir.inference_error or vr.inference_error
    c.inference_result = ir.inference_result or vr.inference_result
    c.parse_result = ir.parse_result or vr.parse_result
    c.video_summaries = vr.video_summaries
    c.errors = ir.errors + vr.errors
    yield c
