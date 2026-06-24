"""Tests for the SpeciesNet result parser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.result_parser import (
    ParseResult,
    parse_entry,
    parse_speciesnet_output,
    _split_taxonomy,
)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "speciesnet_sample_output.json"


def test_parse_blank():
    pr = parse_speciesnet_output(SAMPLE)
    assert pr.success
    blank = [p for p in pr.predictions if "blank" in p.filepath][0]
    assert blank.prediction_label == "blank"
    assert blank.prediction_score == 0.98
    assert len(blank.detections) == 0


def test_parse_animal():
    pr = parse_speciesnet_output(SAMPLE)
    animal = [p for p in pr.predictions if "animal" in p.filepath][0]
    assert animal.prediction_label == "pan troglodytes"
    assert animal.prediction_score == 0.88
    assert len(animal.detections) == 1
    d = animal.detections[0]
    assert d.detector_label == "animal"
    assert d.detector_confidence == 0.95
    assert d.bbox_x == 0.1 and d.bbox_w == 0.4
    assert d.bbox_format == "normalized_xywh"


def test_parse_human():
    pr = parse_speciesnet_output(SAMPLE)
    human = [p for p in pr.predictions if "human" in p.filepath][0]
    assert human.prediction_label == "human"
    assert human.detections[0].detector_label == "human"


def test_parse_multiple_detections():
    pr = parse_speciesnet_output(SAMPLE)
    multi = [p for p in pr.predictions if "multi" in p.filepath][0]
    assert len(multi.detections) == 3
    labels = {d.detector_label for d in multi.detections}
    assert labels == {"animal", "vehicle"}


def test_parse_failure():
    pr = parse_speciesnet_output(SAMPLE)
    fail = [p for p in pr.predictions if "failure" in p.filepath][0]
    assert fail.has_error
    assert "DETECTOR" in fail.failures


def test_parse_vehicle():
    pr = parse_speciesnet_output(SAMPLE)
    veh = [p for p in pr.predictions if "vehicle" in p.filepath][0]
    assert veh.prediction_label == "vehicle"
    assert veh.detections[0].detector_label == "vehicle"


def test_taxonomy_split():
    common, sci, level = _split_taxonomy("pan troglodytes")
    assert sci == "pan troglodytes"
    assert level == "species"
    common, sci, level = _split_taxonomy("mammalia")
    assert level == "class"
    common, sci, level = _split_taxonomy("felidae")
    assert level == "family"


def test_robust_to_missing_fields():
    entry = {"filepath": "x.jpg"}
    p = parse_entry(entry)
    assert p.filepath == "x.jpg"
    assert p.detections == []
    assert p.prediction_label is None
    assert not p.has_error


def test_missing_file():
    pr = parse_speciesnet_output("nonexistent.json")
    assert not pr.success
    assert pr.error is not None


def test_total_predictions():
    pr = parse_speciesnet_output(SAMPLE)
    assert len(pr.predictions) == 7
