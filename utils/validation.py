"""PrimateScope AI — input validation and metadata extraction helpers.

All functions are pure and side-effect free so they are trivially unit-testable.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

# Pattern: optional prefix separator, then CS-1 / CS_1 / STN-ALPHA / Station01 / camera01
_STATION_PATTERN = re.compile(
    r"(?:^|[/_\-])((?:CS|STN|STATION|CAM|CAMERA)[\-_]?[A-Z0-9]+)",
    re.IGNORECASE,
)


def safe_filename(name: str, max_length: int = 120) -> str:
    """Return a filesystem-safe version of *name* preserving the extension.

    Keeps unicode letters/digits, dots, dashes, underscores. Replaces everything
    else (spaces, slashes, control chars, shell metacharacters) with ``_``.
    Truncates the stem without touching the extension.
    """
    if not name:
        return "unnamed"
    name = name.strip()
    # Preserve the extension separately.
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    else:
        ext = "." + ext
    # Sanitize stem: allow word chars and dashes; collapse the rest to _.
    stem = re.sub(r"[^\w\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "file"
    # Sanitize extension (letters only, short).
    ext = re.sub(r"[^\w]", "", ext).lower()
    if ext:
        ext = "." + ext
    if len(stem) > max_length:
        stem = stem[:max_length]
    return f"{stem}{ext}"


def unique_path(dest_dir: Path, filename: str) -> Path:
    """Return a path in *dest_dir* that does not exist, appending a short suffix
    if *filename* already collides. Never overwrites existing files.
    """
    dest_dir = Path(dest_dir)
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    import uuid

    for _ in range(20):
        short = uuid.uuid4().hex[:6]
        candidate = dest_dir / f"{stem}_{short}{suffix}"
        if not candidate.exists():
            return candidate
    # Extremely unlikely fallback.
    candidate = dest_dir / f"{stem}_{uuid.uuid4().hex}{suffix}"
    return candidate


def validate_country_code(code: Optional[str]) -> Optional[str]:
    """Validate and normalize an ISO 3166-1 alpha-3 country code.

    Returns the upper-cased 3-letter code, or None when blank/invalid.
    """
    if not code:
        return None
    code = code.strip().upper()
    if len(code) == 3 and code.isalpha():
        return code
    return None


def get_media_type(filename: str) -> Optional[str]:
    """Return 'image' or 'video' based on extension, else None."""
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def is_supported_media(filename: str) -> bool:
    return get_media_type(filename) is not None


def extract_station_id(filename: str) -> Optional[str]:
    """Best-effort extraction of a station ID from a filename.

    Matches patterns like CS-1, CS_1, STN-ALPHA, Station01, camera01.
    Returns the upper-cased matched token without the leading separator, or None.
    """
    if not filename:
        return None
    m = _STATION_PATTERN.search(filename)
    if m:
        token = m.group(1)
        token = re.sub(r"[\s]+", "", token)
        return token.upper()
    return None


def extract_captured_at(path: Path) -> Optional[str]:
    """Try to read capture timestamp from EXIF (images). Returns ISO string or None.

    Never raises — EXIF parsing is best-effort.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import Base as ExifBase  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            # DateTimeOriginal tag id 36867.
            dt = exif.get(36867) or exif.get(ExifBase.DateTime)
            if dt:
                # Common EXIF format: "2026:04:08 07:42:11"
                dt = dt.strip()
                if ":" in dt[:5]:
                    dt = dt.replace(":", "-", 2)
                return dt
    except Exception:
        return None
    return None


def iso_now() -> str:
    """Current UTC timestamp in ISO-8601 string form."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_checksum(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """Return a sha256 hex digest for *path*, or None on failure."""
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None
