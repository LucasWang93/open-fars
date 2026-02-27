#!/usr/bin/env python3
"""Standalone entry point executed inside an sbatch job on a GPU node.

Usage (typically invoked by sbatch_gen.py):
    python -m src.compute.run_experiment_standalone <project_dir>
"""

import json
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.experiment import run_experiment  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("usage: run_experiment_standalone.py <project_dir>")
        sys.exit(1)
    project_dir = Path(sys.argv[1])
    if not project_dir.is_dir():
        print(f"ERROR: project dir not found: {project_dir}")
        sys.exit(2)

    idea_path = project_dir / "00_idea" / "idea.json"
    plan_path = project_dir / "01_plan" / "plan.md"
    results_dir = project_dir / "02_runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    idea = json.loads(idea_path.read_text())

    raw_results = run_experiment(idea, str(results_dir))

    out = results_dir / "metrics.json"
    out.write_text(json.dumps(raw_results, indent=2, default=str))
    print(f"Experiment done. Results -> {out}")


if __name__ == "__main__":
    main()
