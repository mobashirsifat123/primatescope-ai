"""PrimateScope AI — file storage for uploaded camera-trap media.

All uploaded files are copied into project-specific directories under
``uploads/<project_id>/``. Originals are never overwritten; collisions get a
short unique suffix. Paths are constructed with pathlib and never via shell
concatenation.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from utils.constants import (
    EXPORTS_DIR,
    LOG_DIR,
    MAX_FILE_SIZE_BYTES,
    OUTPUTS_DIR,
    UPLOADS_DIR,
)
from utils.logging_config import get_logger
from utils.validation import (
    extract_captured_at,
    extract_station_id,
    file_checksum,
    get_media_type,
    iso_now,
    safe_filename,
    unique_path,
)

_log = get_logger("file_storage")


class FileStorage:
    """Manage on-disk storage of media, outputs, exports, and logs."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)
        self.uploads = self.root / UPLOADS_DIR
        self.outputs = self.root / OUTPUTS_DIR
        self.exports = self.root / EXPORTS_DIR
        self.logs = self.root / LOG_DIR

    # -- directory scaffolding ------------------------------------------------
    def ensure_project_dirs(self, project_id: str) -> dict[str, Path]:
        """Create and return all project-specific directories."""
        base = self.uploads / project_id
        dirs = {
            "originals": base / "originals",
            "frames": base / "frames",
            "outputs": self.outputs / project_id,
            "exports": self.exports / project_id,
            "logs": self.logs / project_id,
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def originals_dir(self, project_id: str) -> Path:
        d = self.uploads / project_id / "originals"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def frames_dir(self, project_id: str, video_stem: str) -> Path:
        d = self.uploads / project_id / "frames" / video_stem
        d.mkdir(parents=True, exist_ok=True)
        return d

    def outputs_dir(self, project_id: str) -> Path:
        d = self.outputs / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def exports_dir(self, project_id: str) -> Path:
        d = self.exports / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- saving uploads -------------------------------------------------------
    def save_upload(
        self,
        project_id: str,
        src_path: str | Path,
        original_name: str,
    ) -> dict:
        """Copy an uploaded file into the project originals folder.

        Returns a dict with stored_path, original_filename, media_type,
        file_size_bytes, station_id, captured_at, checksum. Raises ValueError
        for unsupported types or oversized files.
        """
        src = Path(src_path)
        media_type = get_media_type(original_name)
        if media_type is None:
            raise ValueError(
                f"Unsupported file type: {original_name}. "
                "Supported: jpg, jpeg, png, bmp, tif, tiff, mp4, mov, avi, mkv."
            )
        size = src.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {original_name} ({size / 1e6:.1f} MB). "
                f"Max {MAX_FILE_SIZE_BYTES / 1e6:.0f} MB."
            )
        dest_dir = self.originals_dir(project_id)
        safe_name = safe_filename(original_name)
        dest = unique_path(dest_dir, safe_name)
        shutil.copy2(str(src), str(dest))
        checksum = file_checksum(dest)
        station_id = extract_station_id(original_name)
        captured_at = None
        if media_type == "image":
            captured_at = extract_captured_at(dest)
        _log.info(
            "Saved upload %s -> %s (%d bytes, %s)",
            original_name, dest, size, media_type,
        )
        return {
            "stored_path": str(dest),
            "original_filename": original_name,
            "media_type": media_type,
            "file_size_bytes": size,
            "station_id": station_id,
            "captured_at": captured_at,
            "checksum": checksum,
        }

    def save_stream(
        self,
        project_id: str,
        file_bytes: bytes,
        original_name: str,
    ) -> dict:
        """Save bytes from a Streamlit UploadedFile via a temp path."""
        import tempfile

        suffix = Path(original_name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return self.save_upload(project_id, tmp_path, original_name)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
