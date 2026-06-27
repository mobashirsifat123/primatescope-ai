"""PrimateScope AI — SpeciesNet inference runner and environment checks.

Wraps the SpeciesNet CLI (``python -m speciesnet.scripts.run_model``) via
``subprocess.run`` with argument lists only (never ``shell=True``). Captures
stdout/stderr/returncode/duration and never raises on inference failure — it
returns a structured result so the Streamlit UI can show a readable error.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

_log = get_logger("speciesnet_runner")

SPECIESNET_MODULE = "speciesnet.scripts.run_model"
MD_MODULE = "megadetector.detection.run_md_and_speciesnet"


def _safe_env() -> dict:
    """Environment for subprocesses that prevents the project root ``utils``
    package from shadowing yolov5's bundled ``utils`` (used by SpeciesNet).

    yolov5 uses flat imports such as ``from utils.autoanchor import ...``.
    Setting ``PYTHONSAFEPATH=1`` stops Python from prepending the current
    working directory to ``sys.path``, avoiding the namespace collision.
    """
    env = dict(os.environ)
    env["PYTHONSAFEPATH"] = "1"
    return env


@dataclass
class InferenceRunResult:
    success: bool
    return_code: int
    output_json_path: Optional[str]
    stdout: str
    stderr: str
    error_message: Optional[str]
    duration_seconds: float
    started_at: str
    finished_at: str
    command: str

    def to_dict(self):
        return asdict(self)


def _run_module_help(module: str, timeout: float = 30.0) -> tuple[bool, str]:
    """Run ``<python> -m <module> --help`` and return (ok, output).

    Some CLIs (e.g. SpeciesNet via abseil) print full usage but exit non-zero
    for ``--help``; treat recognizable usage output as success too.
    """
    cmd = [sys.executable, "-m", module, "--help"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            check=False, env=_safe_env(),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 or ("usage" in out.lower() or "--help" in out)
        return ok, out
    except FileNotFoundError:
        return False, "python executable not found"
    except subprocess.TimeoutExpired:
        return False, f"--help timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


def _import_check_subprocess(import_name: str, timeout: float = 60.0) -> bool:
    """Import a module in a clean subprocess (``PYTHONSAFEPATH=1``).

    Avoids the project root ``utils`` package shadowing yolov5's bundled
    ``utils`` when importing in-process from the project directory.
    """
    code = f"import {import_name}"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            timeout=timeout, check=False, env=_safe_env(),
        )
        return proc.returncode == 0
    except Exception:
        return False


def check_speciesnet_available() -> tuple[bool, str]:
    """Return (available, message) by running ``python -m speciesnet.scripts.run_model --help``."""
    ok, out = _run_module_help(SPECIESNET_MODULE, timeout=60.0)
    if ok:
        return True, "SpeciesNet CLI available"
    # Softer check: import in a clean subprocess to avoid utils shadowing.
    if _import_check_subprocess("speciesnet"):
        return True, "SpeciesNet package importable (CLI help unavailable)"
    return False, "SpeciesNet not installed. Run: pip install speciesnet"


def check_megadetector_available() -> tuple[bool, str]:
    """Return (available, message) for MegaDetector."""
    ok, out = _run_module_help(MD_MODULE, timeout=60.0)
    if ok:
        return True, "MegaDetector CLI available"
    if _import_check_subprocess("megadetector"):
        return True, "MegaDetector package importable (CLI help unavailable)"
    return False, "MegaDetector not installed. Run: pip install megadetector"


def get_engine_version() -> Optional[str]:
    """Best-effort SpeciesNet version string (clean subprocess import)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c",
             "import speciesnet; print(getattr(speciesnet,'__version__',''))"],
            capture_output=True, text=True, timeout=60, check=False,
            env=_safe_env(),
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
        return None
    except Exception:
        return None


def run_speciesnet_on_folder(
    project_id: str,
    input_folder: str | Path,
    output_json: str | Path,
    country_code: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    timeout: Optional[float] = 1800.0,
) -> InferenceRunResult:
    """Run SpeciesNet on a folder of images and write predictions JSON.

    Uses ``python -m speciesnet.scripts.run_model --folders <dir>
    --predictions_json <out> [--country <CODE>]``. Returns a structured result;
    never raises on inference failure.
    """
    input_folder = Path(input_folder)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Verify CLI availability first.
    available, msg = check_speciesnet_available()
    if not available:
        return InferenceRunResult(
            success=False, return_code=-1, output_json_path=None,
            stdout="", stderr="", error_message=msg,
            duration_seconds=0.0, started_at="", finished_at="",
            command="",
        )

    cmd = [
        sys.executable, "-m", SPECIESNET_MODULE,
        "--folders", str(input_folder),
        "--predictions_json", str(output_json),
    ]
    if country_code:
        cmd += ["--country", country_code]
    if extra_args:
        cmd += list(extra_args)

    started = time.time()
    from utils.validation import iso_now

    started_at = iso_now()
    _log.info("SpeciesNet inference start project=%s cmd=%s", project_id, cmd)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_safe_env(),
        )
    except subprocess.TimeoutExpired as e:
        finished_at = iso_now()
        dur = time.time() - started
        err = f"Inference timed out after {timeout}s"
        _log.error(err)
        return InferenceRunResult(
            success=False, return_code=-1, output_json_path=None,
            stdout=e.stdout or "" if isinstance(e.stdout, str) else "",
            stderr=e.stderr or "" if isinstance(e.stderr, str) else "",
            error_message=err, duration_seconds=dur,
            started_at=started_at, finished_at=finished_at,
            command=" ".join(cmd),
        )
    except Exception as e:
        finished_at = iso_now()
        dur = time.time() - started
        _log.error("SpeciesNet run error: %s", e)
        return InferenceRunResult(
            success=False, return_code=-1, output_json_path=None,
            stdout="", stderr="", error_message=str(e),
            duration_seconds=dur, started_at=started_at,
            finished_at=finished_at, command=" ".join(cmd),
        )

    finished_at = iso_now()
    dur = time.time() - started
    rc = proc.returncode
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    json_exists = output_json.exists()

    success = rc == 0 and json_exists
    error = None
    if not success:
        if rc != 0:
            error = f"SpeciesNet exited with code {rc}."
        elif not json_exists:
            error = "SpeciesNet finished but output JSON was not created."

    _log.info(
        "SpeciesNet inference done project=%s success=%s rc=%s dur=%.1fs",
        project_id, success, rc, dur,
    )

    return InferenceRunResult(
        success=success, return_code=rc,
        output_json_path=str(output_json) if json_exists else None,
        stdout=stdout[-8000:], stderr=stderr[-8000:],
        error_message=error, duration_seconds=round(dur, 2),
        started_at=started_at, finished_at=finished_at,
        command=" ".join(cmd),
    )


def run_md_and_speciesnet(
    project_id: str,
    input_folder: str | Path,
    output_json: str | Path,
    country_code: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    timeout: Optional[float] = 1800.0,
) -> InferenceRunResult:
    """Run MegaDetector + SpeciesNet ensemble on a folder.

    Uses ``python -m megadetector.detection.run_md_and_speciesnet <folder>
    <output_json> [--country <CODE>]``. This path supports multi-detection per
    image and is the recommended engine for video frames. Returns a structured
    result; never raises on inference failure.
    """
    input_folder = Path(input_folder)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    available, msg = check_megadetector_available()
    if not available:
        return InferenceRunResult(
            success=False, return_code=-1, output_json_path=None,
            stdout="", stderr="", error_message=msg,
            duration_seconds=0.0, started_at="", finished_at="",
            command="",
        )

    cmd = [
        sys.executable, "-m", MD_MODULE,
        str(input_folder),
        str(output_json),
    ]
    if country_code:
        cmd += ["--country", country_code]
    if extra_args:
        cmd += list(extra_args)

    started = time.time()
    from utils.validation import iso_now

    started_at = iso_now()
    _log.info("MD+SpeciesNet inference start project=%s cmd=%s", project_id, cmd)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_safe_env(),
        )
    except subprocess.TimeoutExpired as e:
        finished_at = iso_now()
        dur = time.time() - started
        err = f"MD+SpeciesNet timed out after {timeout}s"
        _log.error(err)
        return InferenceRunResult(
            success=False, return_code=-1, output_json_path=None,
            stdout=e.stdout or "" if isinstance(e.stdout, str) else "",
            stderr=e.stderr or "" if isinstance(e.stderr, str) else "",
            error_message=err, duration_seconds=dur,
            started_at=started_at, finished_at=finished_at,
            command=" ".join(cmd),
        )
    except Exception as e:
        finished_at = iso_now()
        dur = time.time() - started
        _log.error("MD+SpeciesNet run error: %s", e)
        return InferenceRunResult(
            success=False, return_code=-1, output_json_path=None,
            stdout="", stderr="", error_message=str(e),
            duration_seconds=dur, started_at=started_at,
            finished_at=finished_at, command=" ".join(cmd),
        )

    finished_at = iso_now()
    dur = time.time() - started
    rc = proc.returncode
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    json_exists = output_json.exists()

    success = rc == 0 and json_exists
    error = None
    if not success:
        if rc != 0:
            error = f"MD+SpeciesNet exited with code {rc}."
        elif not json_exists:
            error = "MD+SpeciesNet finished but output JSON was not created."

    _log.info(
        "MD+SpeciesNet inference done project=%s success=%s rc=%s dur=%.1fs",
        project_id, success, rc, dur,
    )

    return InferenceRunResult(
        success=success, return_code=rc,
        output_json_path=str(output_json) if json_exists else None,
        stdout=stdout[-8000:], stderr=stderr[-8000:],
        error_message=error, duration_seconds=round(dur, 2),
        started_at=started_at, finished_at=finished_at,
        command=" ".join(cmd),
    )
