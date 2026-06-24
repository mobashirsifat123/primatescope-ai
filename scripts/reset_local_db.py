#!/usr/bin/env python3
"""PrimateScope AI — reset the local SQLite database.

Removes data/primatescope.db and recreates an empty schema. Does NOT delete
uploaded media, outputs, or exports — only the database.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import DB_PATH, init_db


def main() -> int:
    confirm = "--yes" in sys.argv
    if not confirm:
        print(f"This will delete: {DB_PATH}")
        ans = input("Type 'yes' to confirm: ").strip().lower()
        if ans != "yes":
            print("Aborted.")
            return 1
    p = Path(DB_PATH)
    if p.exists():
        p.unlink()
        print(f"Deleted {DB_PATH}")
    init_db()
    print(f"Recreated empty database at {DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
