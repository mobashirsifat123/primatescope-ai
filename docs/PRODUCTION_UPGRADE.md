# PrimateScope AI — Production Upgrade Guide

## Overview

PrimateScope AI v1.0 upgrades the v0.2 demo prototype into a production-ready,
local-first camera-trap analysis application. Real camera-trap images and short
videos are processed using Google's SpeciesNet (CameraTrapsAI) ensemble and
MegaDetector, with results persisted in SQLite, human review, and CSV export.

## What Changed

### New Architecture

```
app.py                          ← Streamlit entrypoint (preserves Obsidian Canopy)
production_ui.py                ← Real Inference mode page renderers
├── database/
│   ├── db.py                   ← SQLite connection + schema init
│   ├── models.py               ← Dataclasses + DDL for 9 tables
│   └── repositories.py         ← CRUD repositories + stats
├── services/
│   ├── speciesnet_runner.py    ← SpeciesNet CLI wrapper + env checks
│   ├── video_processor.py      ← Frame extraction + clip aggregation
│   ├── result_parser.py        ← SpeciesNet JSON → normalized records
│   ├── export_service.py       ← CSV export with 31-column schema
│   ├── file_storage.py         ← Safe file persistence
│   ├── bbox_draw.py            ← Bounding-box overlay (PIL)
│   ├── queue_logic.py          ← Review queue reason rules
│   └── pipeline.py             ← End-to-end orchestration
├── utils/
│   ├── constants.py            ← Statuses, file types, colors
│   ├── validation.py           ← Safe filenames, country codes, metadata
│   └── logging_config.py       ← Centralized logging
├── tests/                      ← 23 unit tests (pytest)
├── scripts/                    ← check_environment, create_sample_project, reset_db
└── docs/                       ← Production, SpeciesNet, Validation docs
```

### Database Schema (SQLite)

9 tables: `projects`, `media_files`, `inference_runs`, `detections`,
`species_predictions`, `review_items`, `review_actions`, `exports`,
`app_settings`. Auto-created on first run at `data/primatescope.db`.

### Modes

| Mode | Description |
|---|---|
| **Demo Simulation** | Uses existing YOLOv8n + simulated data (default) |
| **Real Inference** | Runs SpeciesNet/MegaDetector on real uploads, persists to DB |

## Production Workflow

1. **Create/select a project** in the sidebar (Real Inference mode)
2. **Upload images/videos** on Live AI Analysis page
3. **App runs SpeciesNet** on images; extracts frames from videos and runs SpeciesNet on frames
4. **Results parsed** into detections + species predictions
5. **Review items created** for every media file with queue reasons
6. **Review** on the Review Queue page: approve, correct, reject, mark uncertain
7. **Export** reviewed data as CSV on Research Insights & Export page

## Key Design Decisions

- **AI prediction ≠ final label**: `prediction_label` is separate from `final_label`
- **Review status ≠ processing status**: media has its own lifecycle
- **Raw JSON preserved**: `raw_prediction_json` stored for debugging
- **Audit trail**: every review action creates a `review_actions` row
- **No cloud**: all processing is local; no telemetry
- **Scientific honesty**: predictions labeled "AI-assisted", never "final"

## Limitations

- Behavior detection (grooming/chasing/feeding) is NOT implemented — requires a behavior model
- Individual primate ID is NOT implemented — requires re-identification
- Field station map is simulated — requires station metadata
- Validation metrics are NOT calculated — requires ground truth labels
- Custom primate model is a future research module
