"""PrimateScope AI — application-wide constants.

Centralizes statuses, queue reasons, supported file types, and design tokens so
UI and services never drift apart. Values mirror the existing Obsidian Canopy
palette used in app.py.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Paths (relative to project root; resolved by callers with pathlib.Path)
# ---------------------------------------------------------------------------
DB_PATH = "data/primatescope.db"
LOG_DIR = "logs"
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
EXPORTS_DIR = "exports"

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
MODE_DEMO = "Demo Simulation"
MODE_REAL = "Real Inference"

# ---------------------------------------------------------------------------
# Processing statuses (media_files.processing_status)
# ---------------------------------------------------------------------------
PROC_UPLOADED = "uploaded"
PROC_PROCESSING = "processing"
PROC_PROCESSED = "processed"
PROC_FAILED = "failed"
PROC_NEEDS_REVIEW = "needs_review"
PROC_REVIEWED = "reviewed"
PROC_EXPORTED = "exported"

PROCESSING_STATUSES = (
    PROC_UPLOADED,
    PROC_PROCESSING,
    PROC_PROCESSED,
    PROC_FAILED,
    PROC_NEEDS_REVIEW,
    PROC_REVIEWED,
    PROC_EXPORTED,
)

# ---------------------------------------------------------------------------
# Review statuses (review_items.review_status)
# ---------------------------------------------------------------------------
REV_PENDING = "pending"
REV_APPROVED = "approved"
REV_CORRECTED = "corrected"
REV_REJECTED = "rejected"
REV_UNCERTAIN = "uncertain"
REV_FLAGGED = "flagged"
REV_BLANK_CONFIRMED = "blank_confirmed"
REV_HUMAN_CONFIRMED = "human_confirmed"
REV_VEHICLE_CONFIRMED = "vehicle_confirmed"

REVIEW_STATUSES = (
    REV_PENDING,
    REV_APPROVED,
    REV_CORRECTED,
    REV_REJECTED,
    REV_UNCERTAIN,
    REV_FLAGGED,
    REV_BLANK_CONFIRMED,
    REV_HUMAN_CONFIRMED,
    REV_VEHICLE_CONFIRMED,
)

# ---------------------------------------------------------------------------
# Queue reasons (review_items.queue_reason)
# ---------------------------------------------------------------------------
QR_BORDERLINE = "borderline_confidence"
QR_MODEL_REVIEW = "model_prediction_review"
QR_BLANK = "blank"
QR_HUMAN = "human_detected"
QR_VEHICLE = "vehicle_detected"
QR_MULTIPLE = "multiple_species"
QR_MISSING = "missing_prediction"
QR_PARSING = "parsing_issue"
QR_RANDOM_QA = "random_qa_sample"
QR_MANUAL = "manual_flag"
QR_VIDEO_CLIP = "video_clip_summary"

QUEUE_REASONS = (
    QR_BORDERLINE,
    QR_MODEL_REVIEW,
    QR_BLANK,
    QR_HUMAN,
    QR_VEHICLE,
    QR_MULTIPLE,
    QR_MISSING,
    QR_PARSING,
    QR_RANDOM_QA,
    QR_MANUAL,
    QR_VIDEO_CLIP,
)

# Confidence threshold below which a prediction is "borderline".
BORDERLINE_CONFIDENCE = 0.70

# ---------------------------------------------------------------------------
# Inference run statuses
# ---------------------------------------------------------------------------
RUN_RUNNING = "running"
RUN_SUCCESS = "success"
RUN_FAILED = "failed"

# ---------------------------------------------------------------------------
# Media types
# ---------------------------------------------------------------------------
MEDIA_IMAGE = "image"
MEDIA_VIDEO = "video"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv"})

# MegaDetector category code -> human label
MD_CATEGORY_MAP = {
    "1": "animal",
    "2": "human",
    "3": "vehicle",
}

# Labels considered non-animal "blank-like" at the ensemble level.
BLANK_LABELS = frozenset({"blank", "empty", "no_animal"})

# Default country code (ISO 3166-1 alpha-3).
DEFAULT_COUNTRY_CODE = "BGD"

# Common country codes offered in the UI selector.
COMMON_COUNTRY_CODES = (
    "BGD", "CHN", "USA", "GBR", "IND", "IDN", "BRA", "KEN",
    "TZA", "ZAF", "THA", "VNM", "MYS", "PHL", "MMR", "LAO",
)

# Default project name when none is created yet.
DEFAULT_PROJECT_NAME = "PrimateScope Local Project"

# Max upload size per file (bytes). 500 MB keeps short videos viable.
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

# Video frame extraction defaults.
# 2.0s interval = 1 frame every 2 seconds. For a 30s clip that's 15 frames
# instead of 30 — cuts SpeciesNet inference time roughly in half while still
# giving adequate temporal coverage for camera-trap wildlife clips.
FRAME_INTERVAL_SECONDS = 2.0
MAX_FRAMES_PER_CLIP = 30

# ---------------------------------------------------------------------------
# Obsidian Canopy design tokens (mirror app.py; used by bbox overlay)
# ---------------------------------------------------------------------------
CLR_BG = "#0A0F0D"
CLR_SURFACE = "#141C19"
CLR_BORDER = "#1E292B"
CLR_SLATE = "#94A3B8"
CLR_TEAL = "#0adec8"
CLR_AMBER = "#FFB84D"
CLR_WHITE = "#FFFFFF"
CLR_ERROR = "#FFB4AB"

# Bounding box colors by detector label.
BBOX_COLOR_ANIMAL = CLR_TEAL
BBOX_COLOR_HUMAN = CLR_ERROR
BBOX_COLOR_VEHICLE = CLR_SLATE
BBOX_COLOR_LOW_CONF = CLR_AMBER

LOW_CONF_THRESHOLD = 0.40
