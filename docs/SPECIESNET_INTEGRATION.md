# SpeciesNet Integration

## What is SpeciesNet?

[SpeciesNet](https://github.com/google/cameratrapai) is an ensemble of AI models
for classifying wildlife in camera trap images, developed by Google. It combines:

1. **MegaDetector** — an object detector that finds animals, humans, and vehicles
2. **Species classifier** — an EfficientNet V2 M model that classifies 2000+ labels

The ensemble uses heuristics and optional geographic filtering to assign a single
prediction per image.

## Installation

```bash
pip install speciesnet
# macOS (if errors occur):
pip install speciesnet --use-pep517
```

For video and multi-detection support, also install MegaDetector:

```bash
pip install megadetector
```

## CLI Usage (verified from official README)

### Images

```bash
python -m speciesnet.scripts.run_model \
    --folders "path/to/images" \
    --predictions_json "path/to/output.json" \
    --country GBR
```

- `--country` accepts ISO 3166-1 alpha-3 codes (e.g., CHN, BGD, USA, GBR)
- `--admin1_region` (USA only) accepts 2-letter state codes
- Model weights download automatically on first run

### Videos / Multi-detection

```bash
python -m megadetector.detection.run_md_and_speciesnet \
    "path/to/folder" "path/to/output.json" --country USA --state CA
```

This script supports video and classifies every detection (not just the highest
confidence one).

## PrimateScope AI Integration

### Image Pipeline (`services/speciesnet_runner.py`)

- Runs `python -m speciesnet.scripts.run_model` via `subprocess.run` (argument list, no shell)
- Captures stdout, stderr, return code, duration
- Returns `InferenceRunResult` dataclass — never raises on failure
- Verifies CLI availability before running

### Video Pipeline (`services/video_processor.py`)

- Extracts ~1 frame/second from each video (configurable)
- Runs SpeciesNet on extracted frames
- Aggregates frame-level predictions into a clip summary
- Does NOT claim behavior recognition

### Output Format

SpeciesNet produces JSON with a `predictions` array. Each entry contains:

| Field | Description |
|---|---|
| `filepath` | Image path |
| `detections` | List of {category, label, conf, bbox} |
| `classifications` | Top-5 classes + scores |
| `prediction` | Final ensemble label |
| `prediction_score` | Final confidence |
| `model_version` | Model version string |
| `failures` | List of failed components (optional) |

Bounding boxes are normalized `[xmin, ymin, width, height]` in [0, 1].

### Parser (`services/result_parser.py`)

- Robust to missing fields and unexpected labels
- Supports multiple detections per image
- Handles blank, human, vehicle, and animal detections
- Infers taxonomy level heuristically (species, genus, family, order, class)
- Preserves raw JSON for debugging

## Environment Check

```bash
python scripts/check_environment.py
```

Verifies Python version, package availability, and CLI availability.
