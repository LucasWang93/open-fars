import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

import yaml

from .orchestrator.workspace import create_project
from .orchestrator.storage import Storage
from .orchestrator.state_machine import tick
from .utils.log import get_logger


LOGGER = get_logger(__name__)
_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    LOGGER.info("received signal %d, will shut down after current project", signum)
    _SHUTDOWN = True


def load_system_config(repo_root: Path) -> dict:
    config_path = repo_root / "config" / "system.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def git_sync(repo_root: Path, project_id: str, state: str) -> None:
    """Commit project outputs and push to origin."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(repo_root), capture_output=True, check=True)
        msg = "project %s: %s" % (project_id, state)
        subprocess.run(["git", "commit", "-m", msg], cwd=str(repo_root), capture_output=True)
        result = subprocess.run(
            ["git", "push"], cwd=str(repo_root), capture_output=True, text=True
        )
        if result.returncode == 0:
            LOGGER.info("git sync OK for %s", project_id)
        else:
            LOGGER.warning("git push failed: %s", result.stderr.strip())
    except Exception as exc:
        LOGGER.warning("git sync error: %s", exc)


def run_once(repo_root: Path) -> str:
    """Run one full project lifecycle. Returns final state."""
    cfg = load_system_config(repo_root)
    project_root = cfg.get("system", {}).get("project_root", "projects")
    projects_dir = (repo_root / project_root).resolve()
    projects_dir.mkdir(parents=True, exist_ok=True)

    db_path = repo_root / "artifacts" / "fars.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = Storage(db_path)

    result = create_project(projects_dir)
    pid = result["project_id"]
    pdir = Path(result["project_dir"])
    meta = json.loads((pdir / "meta.json").read_text())
    storage.register(pid, result["project_dir"], meta)
    LOGGER.info("created project %s", pid)

    if not storage.try_lock(pid):
        LOGGER.error("failed to lock project %s", pid)
        storage.close()
        return "ABORT"

    state = "IDEA"
    try:
        while True:
            state = tick(pdir, repo_root, storage)
            if state in ("DONE", "ABORT"):
                LOGGER.info("project %s finished: %s", pid, state)
                break
    finally:
        storage.unlock(pid)
    storage.close()

    git_sync(repo_root, pid, state)
    return state


def run_daemon(repo_root: Path, max_projects: int = 0) -> None:
    """Continuous daemon loop. max_projects=0 means infinite."""
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    cfg = load_system_config(repo_root)
    daemon_cfg = cfg.get("daemon", {})
    loop_interval = daemon_cfg.get("loop_interval_seconds", 60)

    count = 0
    while True:
        if _SHUTDOWN:
            LOGGER.info("shutdown requested, exiting daemon")
            break
        if max_projects > 0 and count >= max_projects:
            LOGGER.info("reached max projects (%d), exiting daemon", max_projects)
            break

        count += 1
        LOGGER.info("=== daemon cycle %d (max=%s) ===", count, max_projects or "inf")

        try:
            state = run_once(repo_root)
            LOGGER.info("cycle %d finished with state: %s", count, state)
        except Exception:
            LOGGER.exception("cycle %d crashed, will retry after interval", count)

        if _SHUTDOWN:
            break

        LOGGER.info("sleeping %ds before next cycle...", loop_interval)
        time.sleep(loop_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="FARS MVP daemon")
    parser.add_argument("--once", action="store_true",
                        help="Run one project then exit")
    parser.add_argument("--max-projects", type=int, default=0,
                        help="Max projects to run (0=infinite)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.once:
        run_once(repo_root)
    else:
        run_daemon(repo_root, max_projects=args.max_projects)


if __name__ == "__main__":
    main()
