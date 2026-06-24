"""PrimateScope AI — SQLite connection management and schema initialization.

Uses the stdlib ``sqlite3`` module (no ORM dependency). The database file lives
at ``data/primatescope.db`` and is created automatically on first use.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

from .models import SCHEMA_SQL

_log = get_logger("db")

DB_PATH = Path("data/primatescope.db")


def set_db_path(path: str | Path) -> None:
    """Override the default database path (used by tests)."""
    global DB_PATH
    DB_PATH = Path(path)


def get_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Open a connection with row factory and pragmatic pragmas."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: Optional[str | Path] = None) -> Path:
    """Create the database and all tables if they do not exist. Returns the path."""
    conn = get_connection(db_path)
    try:
        for stmt in SCHEMA_SQL:
            conn.execute(stmt)
        conn.commit()
        _log.info("Database initialized at %s", DB_PATH)
    finally:
        conn.close()
    return DB_PATH


def db_exists() -> bool:
    return DB_PATH.exists()
