#!/usr/bin/env python3
"""PrimateScope AI — environment check script.

Prints Python version, package availability, Streamlit version, OpenCV
availability, SpeciesNet CLI availability, and MegaDetector availability.
Errors hard if Python >= 3.14.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging_config import get_logger  # noqa: F401

_log = get_logger("check_env")


def _ok(msg):
    print(f"  [OK]   {msg}")


def _warn(msg):
    print(f"  [WARN] {msg}")


def _err(msg):
    print(f"  [ERR]  {msg}")


def check_python_version() -> bool:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    print(f"\nPython version: {version}")
    if v.major == 3 and v.minor >= 14:
        _err(f"Python {version} is NOT supported. Use Python 3.11 or 3.12.")
        _err("SpeciesNet/MegaDetector may not install on Python 3.14.")
        _err("Install Python 3.12: https://www.python.org/downloads/")
        return False
    if v.major == 3 and v.minor < 11:
        _warn(f"Python {version} is older than recommended 3.11/3.12.")
    else:
        _ok(f"Python {version} is supported (3.11/3.12 recommended).")
    return True


def check_package(name: str, import_name: str | None = None) -> bool:
    imp = import_name or name
    try:
        mod = __import__(imp)
        ver = getattr(mod, "__version__", "unknown")
        _ok(f"{name} {ver}")
        return True
    except Exception:
        _warn(f"{name} not installed")
        return False


def check_package_subprocess(name: str, import_name: str | None = None) -> bool:
    """Import-check a package in a clean subprocess with ``PYTHONSAFEPATH=1``.

    Used for packages (e.g. speciesnet) whose dependencies use flat imports
    that collide with this project's top-level ``utils`` package when the
    project root is on ``sys.path``.
    """
    import os
    import subprocess

    imp = import_name or name
    env = dict(os.environ)
    env["PYTHONSAFEPATH"] = "1"
    code = (
        f"import {imp} as m; "
        f"print(getattr(m, '__version__', 'unknown'))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120, check=False, env=env,
        )
        if proc.returncode == 0:
            _ok(f"{name} {proc.stdout.strip() or 'unknown'}")
            return True
        _warn(f"{name} not installed")
        return False
    except Exception:
        _warn(f"{name} not installed")
        return False


def main() -> int:
    print("=" * 60)
    print("  PrimateScope AI — Environment Check")
    print("=" * 60)

    py_ok = check_python_version()

    print("\nCore packages:")
    check_package("streamlit")
    check_package("pandas")
    check_package("numpy")
    check_package("Pillow", "PIL")
    check_package("opencv-python", "cv2")
    check_package("matplotlib")
    check_package("networkx")
    check_package("SQLAlchemy", "sqlalchemy")

    print("\nML / Inference packages:")
    check_package("ultralytics")
    check_package("torch")
    check_package_subprocess("speciesnet")
    check_package_subprocess("megadetector")

    print("\nCLI availability:")
    from services.speciesnet_runner import (
        check_megadetector_available,
        check_speciesnet_available,
    )
    sn_ok, sn_msg = check_speciesnet_available()
    if sn_ok:
        _ok(f"SpeciesNet CLI: {sn_msg}")
    else:
        _warn(f"SpeciesNet CLI: {sn_msg}")
        _warn("  Install: pip install speciesnet  (macOS: --use-pep517)")

    md_ok, md_msg = check_megadetector_available()
    if md_ok:
        _ok(f"MegaDetector CLI: {md_msg}")
    else:
        _warn(f"MegaDetector CLI: {md_msg}")
        _warn("  Install: pip install megadetector")

    print("\n" + "=" * 60)
    if not py_ok:
        print("  RESULT: BLOCKED — Python version unsupported.")
        print("  Fix: install Python 3.11 or 3.12 and recreate the venv.")
        return 1
    if sn_ok or md_ok:
        print("  RESULT: READY — real inference is available.")
    else:
        print("  RESULT: PARTIAL — app runs in Demo mode.")
        print("  Real inference requires: pip install speciesnet megadetector")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
