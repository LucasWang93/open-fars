#!/usr/bin/env python3
"""Standalone experiment runner - executed on GPU node via sbatch."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.experiment import run_experiment
from src.utils.log import get_logger

LOGGER = get_logger("standalone_experiment")


def main():
    if len(sys.argv) < 3:
        print("Usage: run_experiment_standalone.py <project_dir> <repo_root>")
        sys.exit(1)

    project_dir = Path(sys.argv[1])
    repo_root = Path(sys.argv[2])

    LOGGER.info("starting experiment: project=%s", project_dir.name)
    try:
        run_experiment(project_dir, repo_root)
        marker = project_dir / "02_exp" / ".done"
        marker.write_text("SUCCESS")
        LOGGER.info("experiment completed successfully")
    except Exception as exc:
        LOGGER.exception("experiment failed: %s", exc)
        marker = project_dir / "02_exp" / ".done"
        marker.write_text("FAIL: " + str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
