import argparse
import json
import time
from pathlib import Path

import yaml

from .orchestrator.workspace import create_project
from .orchestrator.storage import Storage
from .orchestrator.state_machine import tick
from .utils.log import get_logger


LOGGER = get_logger(__name__)


def load_system_config(repo_root: Path) -> dict:
    config_path = repo_root / "config" / "system.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def run_once(repo_root: Path) -> None:
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
        return

    try:
        while True:
            state = tick(pdir, repo_root, storage)
            if state in ("DONE", "ABORT"):
                LOGGER.info("project %s finished: %s", pid, state)
                break
    finally:
        storage.unlock(pid)
    storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="FARS MVP daemon")
    parser.add_argument("--once", action="store_true", help="Run one full project cycle")
    parser.add_argument("--max-projects", type=int, default=1)
    parser.add_argument("--sleep", type=int, default=60)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.once:
        run_once(repo_root)
        return

    for i in range(args.max_projects):
        LOGGER.info("=== daemon cycle %d/%d ===", i + 1, args.max_projects)
        run_once(repo_root)
        if i < args.max_projects - 1:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
