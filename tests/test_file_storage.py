"""Tests for file storage utilities and the FileStorage service."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.validation import safe_filename, unique_path, get_media_type, file_checksum


# ---------------------------------------------------------------------------
# safe_filename
# ---------------------------------------------------------------------------
class TestSafeFilename:
    def test_simple_name(self):
        assert safe_filename("hello.jpg") == "hello.jpg"

    def test_spaces_replaced(self):
        result = safe_filename("camera trap 01.png")
        assert " " not in result
        assert result.endswith(".png")

    def test_special_characters(self):
        result = safe_filename("photo/../../../etc/passwd.jpg")
        assert "/" not in result
        assert ".." not in result.replace(".jpg", "")
        assert result.endswith(".jpg")

    def test_empty_string(self):
        assert safe_filename("") == "unnamed"

    def test_long_stem_truncated(self):
        long_name = "a" * 200 + ".tiff"
        result = safe_filename(long_name)
        stem = result.rsplit(".", 1)[0]
        assert len(stem) <= 120

    def test_unicode_preserved(self):
        result = safe_filename("トラップ_01.jpg")
        assert result.endswith(".jpg")
        assert len(result) > 4  # Not just ".jpg"


# ---------------------------------------------------------------------------
# unique_path
# ---------------------------------------------------------------------------
class TestUniquePath:
    def test_no_collision(self, tmp_path):
        p = unique_path(tmp_path, "image.jpg")
        assert p == tmp_path / "image.jpg"

    def test_collision_resolved(self, tmp_path):
        (tmp_path / "image.jpg").write_text("existing")
        p = unique_path(tmp_path, "image.jpg")
        assert p != tmp_path / "image.jpg"
        assert p.suffix == ".jpg"
        assert not p.exists()

    def test_multiple_collisions(self, tmp_path):
        (tmp_path / "file.png").write_text("1")
        p1 = unique_path(tmp_path, "file.png")
        p1.write_text("2")
        p2 = unique_path(tmp_path, "file.png")
        assert p1 != p2
        assert p2.suffix == ".png"


# ---------------------------------------------------------------------------
# get_media_type
# ---------------------------------------------------------------------------
class TestGetMediaType:
    @pytest.mark.parametrize("name,expected", [
        ("photo.jpg", "image"),
        ("photo.JPEG", "image"),
        ("photo.png", "image"),
        ("photo.bmp", "image"),
        ("photo.tif", "image"),
        ("photo.tiff", "image"),
        ("clip.mp4", "video"),
        ("clip.MOV", "video"),
        ("clip.avi", "video"),
        ("clip.mkv", "video"),
    ])
    def test_supported_types(self, name, expected):
        assert get_media_type(name) == expected

    @pytest.mark.parametrize("name", ["file.txt", "data.csv", "doc.pdf", ""])
    def test_unsupported_types(self, name):
        assert get_media_type(name) is None


# ---------------------------------------------------------------------------
# file_checksum
# ---------------------------------------------------------------------------
class TestFileChecksum:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello camera trap")
        c1 = file_checksum(f)
        c2 = file_checksum(f)
        assert c1 is not None
        assert c1 == c2
        assert len(c1) == 64  # sha256 hex digest

    def test_different_content(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"data A")
        f2.write_bytes(b"data B")
        assert file_checksum(f1) != file_checksum(f2)

    def test_missing_file(self, tmp_path):
        assert file_checksum(tmp_path / "nonexistent.bin") is None


# ---------------------------------------------------------------------------
# FileStorage.save_stream
# ---------------------------------------------------------------------------
class TestFileStorageSaveStream:
    def test_save_and_read(self, tmp_path):
        from services.file_storage import FileStorage
        storage = FileStorage(root=tmp_path)
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # fake PNG header
        info = storage.save_stream("proj_test", data, "test_image.png")
        assert info["original_filename"] == "test_image.png"
        assert info["media_type"] == "image"
        assert Path(info["stored_path"]).exists()
        assert info["file_size_bytes"] > 0
        assert info["checksum"] is not None

    def test_unsupported_type_raises(self, tmp_path):
        from services.file_storage import FileStorage
        storage = FileStorage(root=tmp_path)
        with pytest.raises(ValueError, match="Unsupported"):
            storage.save_stream("proj_test", b"data", "readme.txt")

    def test_no_overwrite(self, tmp_path):
        from services.file_storage import FileStorage
        storage = FileStorage(root=tmp_path)
        data = b"\x89PNG" + b"\x00" * 50
        info1 = storage.save_stream("proj_test", data, "photo.png")
        info2 = storage.save_stream("proj_test", data, "photo.png")
        # Second save should not overwrite the first.
        assert info1["stored_path"] != info2["stored_path"]
        assert Path(info1["stored_path"]).exists()
        assert Path(info2["stored_path"]).exists()
