"""PrimateScope AI — SpeciesNet/MegaDetector JSON output parser.

Handles TWO output formats:
  1. SpeciesNet ``run_model`` → ``{"predictions": [...]}`` with filepath,
     prediction, prediction_score, model_version per image.
  2. MegaDetector ``run_md_and_speciesnet`` → ``{"images": [...]}`` with file,
     detections (each carrying classifications [[idx, score], ...]),
     plus top-level detection_categories and classification_categories dicts.

Robust to missing fields, unexpected labels, and parse failures. Never raises
on a single bad entry — records a warning and continues.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from utils.constants import MD_CATEGORY_MAP
from utils.logging_config import get_logger

_log = get_logger("result_parser")


@dataclass
class ParsedDetection:
    detector_label: Optional[str] = None
    detector_confidence: Optional[float] = None
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_w: Optional[float] = None
    bbox_h: Optional[float] = None
    bbox_format: str = "normalized_xywh"


@dataclass
class ParsedPrediction:
    filepath: str
    detections: list[ParsedDetection] = field(default_factory=list)
    prediction_label: Optional[str] = None
    prediction_score: Optional[float] = None
    prediction_source: Optional[str] = None
    common_name: Optional[str] = None
    scientific_name: Optional[str] = None
    taxonomy_level: Optional[str] = None
    model_version: Optional[str] = None
    failures: list[str] = field(default_factory=list)
    has_error: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return Path(self.filepath).name


@dataclass
class ParseResult:
    predictions: list[ParsedPrediction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


def load_json(path: str | Path) -> Optional[dict]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        _log.error("Output JSON not found: %s", path)
        return None
    except json.JSONDecodeError as e:
        _log.error("JSON parse error in %s: %s", path, e)
        return None
    except Exception as e:
        _log.error("Unexpected error reading %s: %s", path, e)
        return None


def _infer_level(token: str) -> Optional[str]:
    """Heuristic taxonomy level inference."""
    t = token.lower().strip()
    if not t:
        return None
    suffix_map = {
        "kingdom": ["animalia", "plantae"],
        "class": ["mammalia", "aves", "reptilia", "amphibia", "insecta"],
        "order": ["primates", "carnivora", "rodentia", "artiodactyla",
                   "cetacea", "chiroptera"],
        "family": ["felidae", "cercopithecidae", "hylobatidae", "hominidae",
                    "cervidae", "bovidae", "ursidae", "canidae"],
        "genus": ["pan", "gorilla", "pongo", "macaca", "papio"],
    }
    for level, tokens in suffix_map.items():
        if t in tokens:
            return level
    if " " in t and t.split()[0][0].islower():
        return "species"
    if t.endswith("idae"):
        return "family"
    if t.endswith("formes"):
        return "order"
    return None


def _split_taxonomy(label: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Split a label into (common_name, scientific_name, taxonomy_level)."""
    if not label:
        return None, None, None
    raw = label.strip()
    common = None
    scientific = None
    if "(" in raw and ")" in raw:
        inner = raw[raw.find("(") + 1: raw.rfind(")")]
        outer = raw[: raw.find("(")].strip()
        common = inner
        scientific = outer
    elif " " in raw and raw[0].islower():
        scientific = raw
    elif raw.islower() or raw.isupper():
        scientific = raw
    else:
        common = raw
    level = _infer_level(scientific or common or raw)
    return common, scientific, level


def _parse_detection(d: dict) -> ParsedDetection:
    """Parse a single detection dict (SpeciesNet run_model format)."""
    label = d.get("label")
    if not label:
        cat = str(d.get("category", ""))
        label = MD_CATEGORY_MAP.get(cat, cat or None)
    conf = d.get("conf")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    bbox = d.get("bbox") or []
    bx = by = bw = bh = None
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            bx, by, bw, bh = (float(v) for v in bbox)
        except (TypeError, ValueError):
            bx = by = bw = bh = None
    return ParsedDetection(
        detector_label=label, detector_confidence=conf,
        bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
        bbox_format="normalized_xywh",
    )


def parse_entry(entry: dict) -> ParsedPrediction:
    """Parse a single prediction entry (SpeciesNet run_model format)."""
    filepath = entry.get("filepath", "")
    failures = entry.get("failures") or []
    has_error = bool(failures)
    detections = []
    for d in entry.get("detections") or []:
        try:
            detections.append(_parse_detection(d))
        except Exception as e:
            _log.warning("Bad detection in %s: %s", filepath, e)
    pred_label = entry.get("prediction")
    pred_score = entry.get("prediction_score")
    try:
        pred_score = float(pred_score) if pred_score is not None else None
    except (TypeError, ValueError):
        pred_score = None
    common, scientific, level = _split_taxonomy(pred_label or "")
    return ParsedPrediction(
        filepath=filepath, detections=detections,
        prediction_label=pred_label, prediction_score=pred_score,
        prediction_source=entry.get("prediction_source"),
        common_name=common, scientific_name=scientific,
        taxonomy_level=level,
        model_version=entry.get("model_version"),
        failures=list(failures), has_error=has_error,
        raw=entry,
    )


# ---------------------------------------------------------------------------
# MegaDetector format (run_md_and_speciesnet)
# ---------------------------------------------------------------------------

def _parse_md_detection(
    d: dict,
    det_categories: dict[str, str],
    cls_categories: dict[str, str],
    cls_descriptions: dict[str, str],
) -> tuple[ParsedDetection, Optional[str], Optional[float]]:
    """Parse a MegaDetector-format detection.

    Returns (detection, best_species_label, best_species_score).
    """
    cat = str(d.get("category", ""))
    label = det_categories.get(cat, MD_CATEGORY_MAP.get(cat, cat or None))
    # Normalize "person" → "human" for consistency.
    if label == "person":
        label = "human"
    conf = d.get("conf")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    bbox = d.get("bbox") or []
    bx = by = bw = bh = None
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            bx, by, bw, bh = (float(v) for v in bbox)
        except (TypeError, ValueError):
            bx = by = bw = bh = None
    det = ParsedDetection(
        detector_label=label, detector_confidence=conf,
        bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
        bbox_format="normalized_xywh",
    )
    # Find best classification for this detection.
    classifications = d.get("classifications") or []
    best_label = None
    best_score = None
    for cls_entry in classifications:
        if not isinstance(cls_entry, list) or len(cls_entry) < 2:
            continue
        cls_idx = str(cls_entry[0])
        cls_score = cls_entry[1]
        try:
            cls_score = float(cls_score)
        except (TypeError, ValueError):
            continue
        if best_score is None or cls_score > best_score:
            best_score = cls_score
            best_label = cls_categories.get(cls_idx, None)
    return det, best_label, best_score


def _parse_md_entry(
    entry: dict,
    det_categories: dict[str, str],
    cls_categories: dict[str, str],
    cls_descriptions: dict[str, str],
) -> ParsedPrediction:
    """Parse a single MegaDetector-format image entry."""
    filepath = entry.get("file") or entry.get("filepath", "")
    failures = []
    if entry.get("error"):
        failures.append(entry["error"])
    has_error = bool(failures)
    detections = []
    best_species_label = None
    best_species_score = None
    for d in entry.get("detections") or []:
        try:
            det, sp_label, sp_score = _parse_md_detection(
                d, det_categories, cls_categories, cls_descriptions
            )
            detections.append(det)
            # Track the best species classification across all animal detections.
            if det.detector_label == "animal" and sp_score is not None:
                if best_species_score is None or sp_score > best_species_score:
                    best_species_score = sp_score
                    best_species_label = sp_label
        except Exception as e:
            _log.warning("Bad MD detection in %s: %s", filepath, e)
    # For MegaDetector format, the "prediction" is the best species
    # classification from the highest-confidence animal detection.
    # If no animal detections, use the detector label (human/vehicle/blank).
    pred_label = best_species_label
    pred_score = best_species_score
    if pred_label is None and detections:
        # Use the top detection label as the prediction.
        top = max(detections, key=lambda x: x.detector_confidence or 0)
        pred_label = top.detector_label
        pred_score = top.detector_confidence
    if pred_label is None and not detections:
        pred_label = "blank"
        pred_score = None
    # Extract taxonomy from description if available.
    common, scientific, level = None, None, None
    if pred_label:
        common, scientific, level = _split_taxonomy(pred_label)
    # Try to get richer taxonomy from classification_category_descriptions.
    if cls_descriptions and best_species_label:
        for idx, name in cls_categories.items():
            if name == best_species_label and idx in cls_descriptions:
                desc = cls_descriptions[idx]
                # Format: "uuid;class;order;family;genus;species;common_name"
                parts = desc.split(";")
                if len(parts) >= 7:
                    if parts[6]:
                        common = parts[6]
                    sci_parts = [p for p in parts[3:6] if p]
                    if sci_parts:
                        scientific = " ".join(sci_parts)
                    if parts[2]:
                        level = _infer_level(parts[2]) or level
                break
    return ParsedPrediction(
        filepath=filepath, detections=detections,
        prediction_label=pred_label, prediction_score=pred_score,
        prediction_source="md_and_speciesnet",
        common_name=common, scientific_name=scientific,
        taxonomy_level=level,
        model_version=None,
        failures=list(failures), has_error=has_error,
        raw=entry,
    )


def parse_speciesnet_output(path: str | Path) -> ParseResult:
    """Parse a SpeciesNet or MegaDetector predictions JSON file.

    Auto-detects the format:
    - ``predictions`` key → SpeciesNet run_model format
    - ``images`` key → MegaDetector run_md_and_speciesnet format
    """
    data = load_json(path)
    if data is None:
        return ParseResult(error=f"Could not read output JSON: {path}")

    # --- SpeciesNet run_model format ---
    if "predictions" in data and isinstance(data["predictions"], list):
        out: list[ParsedPrediction] = []
        warnings: list[str] = []
        for i, entry in enumerate(data["predictions"]):
            if not isinstance(entry, dict):
                warnings.append(f"Entry {i} is not an object; skipped")
                continue
            try:
                out.append(parse_entry(entry))
            except Exception as e:
                warnings.append(f"Entry {i} parse error: {e}")
        _log.info("Parsed %d predictions (SpeciesNet format) from %s", len(out), path)
        return ParseResult(predictions=out, warnings=warnings)

    # --- MegaDetector run_md_and_speciesnet format ---
    if "images" in data and isinstance(data["images"], list):
        det_cats = data.get("detection_categories", {})
        cls_cats = data.get("classification_categories", {})
        cls_descs = data.get("classification_category_descriptions", {})
        out = []
        warnings = []
        for i, entry in enumerate(data["images"]):
            if not isinstance(entry, dict):
                warnings.append(f"Image {i} is not an object; skipped")
                continue
            try:
                out.append(_parse_md_entry(entry, det_cats, cls_cats, cls_descs))
            except Exception as e:
                warnings.append(f"Image {i} parse error: {e}")
        _log.info("Parsed %d predictions (MegaDetector format) from %s", len(out), path)
        return ParseResult(predictions=out, warnings=warnings)

    return ParseResult(error="Output JSON has neither 'predictions' nor 'images' array")


def normalize_bbox_to_pixels(
    bbox_x: float, bbox_y: float, bbox_w: float, bbox_h: float,
    width: int, height: int, bbox_format: str = "normalized_xywh",
) -> tuple[float, float, float, float]:
    """Convert a bbox to absolute pixel xywh coordinates."""
    if bbox_format == "normalized_xywh":
        return (bbox_x * width, bbox_y * height, bbox_w * width, bbox_h * height)
    return bbox_x, bbox_y, bbox_w, bbox_h
