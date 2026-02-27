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

    _record_to_knowledge(repo_root, pdir, pid, state)
    git_sync(repo_root, pid, state)
    return state


def _record_to_knowledge(repo_root: Path, pdir: Path, pid: str, state: str) -> None:
    """Record experiment outcome into the knowledge graph for history tracking."""
    try:
        from .knowledge.kg_store import KnowledgeGraph
        from .knowledge.schemas import ExperimentRecord
        from .utils.time import utc_now

        kc_path = repo_root / "config" / "knowledge.yaml"
        if kc_path.exists():
            kc = yaml.safe_load(kc_path.read_text()).get("knowledge", {})
        else:
            kc = {}
        kg_path = repo_root / kc.get("kg_db_path", "artifacts/knowledge.db")

        idea_path = pdir / "00_idea" / "idea.json"
        if not idea_path.exists():
            return

        idea = json.loads(idea_path.read_text())
        actions = idea.get("actions", [])
        hypothesis = idea.get("hypothesis", "")
        pattern_id = idea.get("pattern_id")

        eval_loss = None
        analysis_path = pdir / "03_results" / "analysis.md"
        if analysis_path.exists():
            import re
            text = analysis_path.read_text()
            m = re.search(r"Treatment: mean=([0-9.]+)", text)
            if m:
                eval_loss = float(m.group(1))

        if state == "DONE":
            outcome = "negative" if eval_loss and eval_loss > 5.0 else "success"
        else:
            outcome = "failed"

        record = ExperimentRecord(
            project_id=pid,
            pattern_id=pattern_id,
            actions=actions,
            hypothesis=hypothesis,
            outcome=outcome,
            eval_loss=eval_loss,
            timestamp=utc_now(),
        )

        kg = KnowledgeGraph(kg_path)
        kg.record_experiment(record)
        kg.close()
        LOGGER.info("recorded experiment history for %s: %s", pid, outcome)
    except Exception as exc:
        LOGGER.warning("failed to record experiment history: %s", exc)


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
            try:
                from .knowledge.stats import log_stats
                log_stats(repo_root)
            except Exception:
                pass
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
