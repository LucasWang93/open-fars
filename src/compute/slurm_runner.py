"""Slurm job submission, polling, and result collection."""

import subprocess
import time
from pathlib import Path

from ..utils.log import get_logger

LOGGER = get_logger(__name__)

DEFAULT_SLURM = {
    "partition": "gpu",
    "gres": "gpu:rtx_5000_ada:4",
    "time": "02:00:00",
    "mem": "120G",
    "cpus_per_task": "16",
    "fallback_partitions": ["scavenge_gpu"],
}

POLL_INTERVAL_INITIAL = 30
POLL_INTERVAL_MAX = 120
POLL_BACKOFF = 1.5


def submit_job(
    script_path: Path,
    job_name: str,
    log_dir: Path,
    slurm_cfg: dict = None,
) -> int:
    """Submit sbatch script. Falls back to alternative partitions on failure."""
    cfg = {**DEFAULT_SLURM, **(slurm_cfg or {})}
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["sbatch", "--parsable", str(script_path)]
    LOGGER.info("submitting: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        err = result.stderr.strip()
        LOGGER.warning("sbatch failed: %s", err)

        fallbacks = cfg.get("fallback_partitions", [])
        for fb_partition in fallbacks:
            LOGGER.info("retrying with partition override: %s", fb_partition)
            retry_cmd = ["sbatch", "--parsable", "-p", fb_partition, str(script_path)]
            result = subprocess.run(retry_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
            LOGGER.warning("fallback %s failed: %s", fb_partition, result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError("sbatch failed on all partitions: " + result.stderr.strip())

    job_id = int(result.stdout.strip().split(";")[0])
    LOGGER.info("submitted job %d", job_id)
    return job_id


def poll_job(job_id: int, timeout_minutes: int = 180) -> str:
    deadline = time.monotonic() + timeout_minutes * 60
    interval = POLL_INTERVAL_INITIAL

    while time.monotonic() < deadline:
        state = _get_job_state(job_id)
        if state is None:
            LOGGER.info("job %d not in squeue, checking sacct...", job_id)
            state = _get_job_state_sacct(job_id)

        if state in ("COMPLETED",):
            LOGGER.info("job %d COMPLETED", job_id)
            return "COMPLETED"
        if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"):
            LOGGER.warning("job %d ended with state: %s", job_id, state)
            return state
        if state in ("COMPLETING",):
            time.sleep(5)
            continue

        LOGGER.info("job %d state=%s, waiting %ds...", job_id, state or "UNKNOWN", int(interval))
        time.sleep(interval)
        interval = min(interval * POLL_BACKOFF, POLL_INTERVAL_MAX)

    LOGGER.error("job %d timed out after %d minutes", job_id, timeout_minutes)
    subprocess.run(["scancel", str(job_id)], capture_output=True)
    return "TIMEOUT"


def _get_job_state(job_id: int):
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True, text=True,
    )
    state = result.stdout.strip()
    return state if state else None


def _get_job_state_sacct(job_id: int):
    result = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=State", "-n", "-P"],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    if lines:
        return lines[0]
    return "COMPLETED"
