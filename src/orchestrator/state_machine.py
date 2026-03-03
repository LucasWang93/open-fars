"""Project state machine: advances one state per tick()."""

import json
from pathlib import Path

import yaml

from ..utils.log import get_logger
from ..utils.time import utc_now
from .gates import gate_a_ideation, gate_b_experiment, gate_c_paper
from .recovery import diagnose_and_recover, run_revise, handle_slurm_failure
from .storage import Storage

LOGGER = get_logger(__name__)

TRANSITIONS = ["IDEA", "PLAN", "RUN", "ANALYZE", "WRITE", "PUBLISH", "DONE"]
MAX_RETRIES = 3
MAX_REVISIONS = 1


def _load_meta(project_dir: Path) -> dict:
    return json.loads((project_dir / "meta.json").read_text())


def _save_meta(project_dir: Path, meta: dict) -> None:
    meta["updated_at"] = utc_now()
    (project_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _load_taskspace(repo_root: Path) -> dict:
    ts_path = repo_root / "config" / "taskspace.yaml"
    return yaml.safe_load(ts_path.read_text())["taskspace"]


def _get_ideation_mode(repo_root: Path) -> str:
    sys_path = repo_root / "config" / "system.yaml"
    if sys_path.exists():
        cfg = yaml.safe_load(sys_path.read_text()) or {}
        return cfg.get("ideation", {}).get("mode", "pattern")
    return "pattern"


def _smart_fail_or_retry(
    project_dir: Path,
    repo_root: Path,
    meta: dict,
    storage: Storage,
    reason: str,
    stage: str,
) -> str:
    """Intelligent failure handling with diagnosis, recovery, and REVISE."""
    pid = meta["project_id"]
    meta["retry_count"] = meta.get("retry_count", 0) + 1

    action = diagnose_and_recover(project_dir, meta, reason, stage)

    if action.retry and meta["retry_count"] <= MAX_RETRIES:
        meta["failure_reason"] = reason
        _save_meta(project_dir, meta)
        storage.update_state(pid, meta["state"], meta)
        LOGGER.warning(
            "project %s retry %d (%s): %s",
            pid, meta["retry_count"], action.strategy, action.description,
        )
        return meta["state"]

    revision_count = meta.get("revision_count", 0)
    if revision_count < MAX_REVISIONS:
        LOGGER.info("project %s: all retries exhausted, attempting REVISE", pid)
        revised = run_revise(project_dir, repo_root)
        if revised:
            meta["revision_count"] = revision_count + 1
            meta["retry_count"] = 0
            meta["state"] = "PLAN"
            meta["failure_reason"] = None
            _save_meta(project_dir, meta)
            storage.update_state(pid, "PLAN", meta)
            LOGGER.info("project %s: REVISED idea, restarting from PLAN", pid)
            return "PLAN"

    meta["state"] = "ABORT"
    meta["failure_reason"] = reason
    _save_meta(project_dir, meta)
    storage.update_state(pid, "ABORT", meta)
    LOGGER.error("project %s ABORT: %s", pid, reason)
    return "ABORT"


def tick(project_dir: Path, repo_root: Path, storage: Storage) -> str:
    """Advance one state. Returns new state string."""
    meta = _load_meta(project_dir)
    state = meta["state"]
    pid = meta["project_id"]

    if state in ("DONE", "ABORT"):
        return state

    LOGGER.info("tick project=%s state=%s", pid, state)

    try:
        if state == "IDEA":
            mode = _get_ideation_mode(repo_root)
            if mode == "pattern":
                from ..agents.ideation_enhanced import run_ideation_enhanced
                run_ideation_enhanced(project_dir, repo_root)
            else:
                from ..agents.ideation import run_ideation
                run_ideation(project_dir, repo_root)

            ok, msg = gate_a_ideation(project_dir)
            if not ok:
                return _smart_fail_or_retry(
                    project_dir, repo_root, meta, storage, f"Gate A: {msg}", "IDEA"
                )
            meta["state"] = "PLAN"

        elif state == "PLAN":
            mode = _get_ideation_mode(repo_root)
            if mode == "pattern":
                from ..agents.planning_enhanced import run_planning_enhanced
                run_planning_enhanced(project_dir, repo_root)
            else:
                from ..agents.planning import run_planning
                run_planning(project_dir, repo_root)
            meta["state"] = "RUN"

        elif state == "RUN":
            from ..compute.sbatch_gen import generate_sbatch_script
            from ..compute.slurm_runner import submit_job, poll_job

            sys_cfg = yaml.safe_load(
                (repo_root / "config" / "system.yaml").read_text()
            )
            slurm_cfg = sys_cfg.get("slurm", {})
            log_dir = repo_root / "artifacts" / "slurm_logs" / pid
            script = generate_sbatch_script(project_dir, repo_root, log_dir, slurm_cfg)

            job_id = submit_job(script, "fars_" + pid, log_dir, slurm_cfg)
            meta["slurm_job_id"] = job_id
            _save_meta(project_dir, meta)
            storage.update_state(pid, "RUN", meta)

            LOGGER.info("experiment submitted as slurm job %d, polling...", job_id)
            job_state = poll_job(
                job_id, timeout_minutes=slurm_cfg.get("timeout_minutes", 180)
            )

            if job_state != "COMPLETED":
                slurm_action = handle_slurm_failure(job_state, project_dir, meta)
                err_log = _read_slurm_error_log(log_dir, job_id)
                reason = f"Slurm job {job_id}: {job_state}"
                if err_log:
                    reason += f"\n{err_log}"
                return _smart_fail_or_retry(
                    project_dir, repo_root, meta, storage, reason, "RUN"
                )

            ok, msg = gate_b_experiment(project_dir)
            if not ok:
                return _smart_fail_or_retry(
                    project_dir, repo_root, meta, storage, f"Gate B: {msg}", "RUN"
                )
            meta["state"] = "ANALYZE"

        elif state == "ANALYZE":
            from ..eval.evaluator import run_evaluation
            run_evaluation(project_dir)
            meta["state"] = "WRITE"

        elif state == "WRITE":
            from ..agents.writing import run_writing
            run_writing(project_dir)
            ok, msg = gate_c_paper(project_dir)
            if not ok:
                return _smart_fail_or_retry(
                    project_dir, repo_root, meta, storage, f"Gate C: {msg}", "WRITE"
                )
            meta["state"] = "PUBLISH"

        elif state == "PUBLISH":
            from ..paper.writer import run_publish
            run_publish(project_dir)
            meta["state"] = "DONE"

    except Exception as exc:
        LOGGER.exception("tick failed for %s at %s", pid, state)
        return _smart_fail_or_retry(
            project_dir, repo_root, meta, storage, str(exc), state
        )

    _save_meta(project_dir, meta)
    storage.update_state(pid, meta["state"], meta)
    LOGGER.info("project %s -> %s", pid, meta["state"])
    return meta["state"]


def _read_slurm_error_log(log_dir: Path, job_id: int, max_chars: int = 2000) -> str:
    """Read the last portion of a Slurm .err log file."""
    err_file = log_dir / f"{job_id}.err"
    if not err_file.exists():
        return ""
    try:
        text = err_file.read_text()
        return text[-max_chars:] if len(text) > max_chars else text
    except Exception:
        return ""
