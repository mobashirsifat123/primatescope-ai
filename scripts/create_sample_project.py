#!/usr/bin/env python3
"""PrimateScope AI — create a sample project with test fixtures.

Creates a project in the database and (optionally) copies sample images so the
review/export flow can be demonstrated without running real inference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import get_connection, init_db
from database.repositories import ProjectRepo
from utils.constants import DEFAULT_COUNTRY_CODE


def main() -> int:
    init_db()
    conn = get_connection()
    try:
        name = sys.argv[1] if len(sys.argv) > 1 else "Sample Field Project"
        country = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_COUNTRY_CODE
        p = ProjectRepo.create(conn, name, "Created by create_sample_project.py", country)
        print(f"Created project: {p.id}")
        print(f"  Name: {p.name}")
        print(f"  Country: {p.country_code}")
        print(f"  Created: {p.created_at}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
