"""PrimateScope AI — centralized logging configuration.

Sets up a root project logger that writes to logs/app.log and stderr.
Callers use ``get_logger(__name__)`` to obtain a configured child logger.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "app.log"
_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the project-wide logger. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("primatescope")
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # File handler.
    try:
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    # Stream handler (stderr) — Streamlit captures stdout/stderr.
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    _CONFIGURED = True


def get_logger(name: str = "primatescope") -> logging.Logger:
    """Return a child logger under the primatescope namespace."""
    if not _CONFIGURED:
        setup_logging()
    if name.startswith("primatescope"):
        return logging.getLogger(name)
    return logging.getLogger(f"primatescope.{name}")
