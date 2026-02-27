#!/usr/bin/env python3
"""Standalone entry point executed inside an sbatch job on a GPU node.

Usage (invoked by sbatch_gen.py):
    python3 src/compute/run_experiment_standalone.py <project_dir> <repo_root>
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("usage: run_experiment_standalone.py <project_dir> <repo_root>")
        sys.exit(1)

    project_dir = Path(sys.argv[1])
    repo_root = Path(sys.argv[2])

    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from src.agents.experiment import run_experiment

    if not project_dir.is_dir():
        print(f"ERROR: project dir not found: {project_dir}")
        sys.exit(2)

    run_experiment(project_dir, repo_root)
    print(f"Experiment done for {project_dir.name}")


if __name__ == "__main__":
    main()
