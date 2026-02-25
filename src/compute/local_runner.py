"""Local subprocess runner with timeout and log capture."""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..utils.log import get_logger

LOGGER = get_logger(__name__)

DEFAULT_TIMEOUT = 120


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[int, str]:
    """Run a command, capture stdout+stderr, enforce timeout.
    Returns (returncode, output_text).
    """
    LOGGER.info("running: %s (cwd=%s, timeout=%ds)", " ".join(cmd), cwd, timeout)
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + proc.stderr
        elapsed = time.monotonic() - start
        LOGGER.info("command finished rc=%d in %.1fs", proc.returncode, elapsed)
    except subprocess.TimeoutExpired:
        output = f"TIMEOUT after {timeout}s"
        LOGGER.error(output)
        log_path.write_text(output)
        return -1, output

    log_path.write_text(output)
    return proc.returncode, output


def run_experiment_script(
    script: str,
    config_path: Path,
    run_dir: Path,
    seed: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[dict]:
    """Run a single experiment seed. Returns metrics dict or None on failure."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    cmd = script.split() + ["--config", str(config_path), "--seed", str(seed), "--output-dir", str(run_dir)]

    rc, output = run_command(cmd, cwd=run_dir.parent.parent, log_path=log_file, timeout=timeout)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            LOGGER.error("metrics.json is invalid JSON in %s", run_dir)
            return None
    LOGGER.warning("no metrics.json produced in %s (rc=%d)", run_dir, rc)
    return None
