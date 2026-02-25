"""Gate checks between pipeline stages."""

import json
from pathlib import Path
from typing import Tuple

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def gate_a_ideation(project_dir: Path, valid_action_ids: list[str]) -> Tuple[bool, str]:
    """Validate idea.md output: chosen actions must be in taskspace."""
    idea_path = project_dir / "00_idea" / "idea.json"
    if not idea_path.exists():
        return False, "idea.json not found"
    try:
        idea = json.loads(idea_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"idea.json parse error: {exc}"

    chosen = idea.get("actions", [])
    if not chosen:
        return False, "no actions chosen"
    for a in chosen:
        if a not in valid_action_ids:
            return False, f"invalid action: {a}"
    return True, "ok"


def gate_b_experiment(project_dir: Path) -> Tuple[bool, str]:
    """Validate that experiment produced valid metrics.json files."""
    runs_dir = project_dir / "02_exp" / "runs"
    if not runs_dir.exists():
        return False, "runs directory missing"
    run_dirs = sorted(runs_dir.iterdir())
    if not run_dirs:
        return False, "no run directories"

    for rd in run_dirs:
        mf = rd / "metrics.json"
        if not mf.exists():
            return False, f"metrics.json missing in {rd.name}"
        try:
            m = json.loads(mf.read_text())
        except json.JSONDecodeError as exc:
            return False, f"metrics.json parse error in {rd.name}: {exc}"
        if m.get("status") not in ("SUCCESS", "FAIL"):
            return False, f"invalid status in {rd.name}"
        if m.get("status") == "SUCCESS" and "primary_metric" not in m:
            return False, f"missing primary_metric in {rd.name}"
    return True, "ok"


def gate_c_paper(project_dir: Path) -> Tuple[bool, str]:
    """Validate paper.md completeness."""
    paper_path = project_dir / "04_paper" / "paper.md"
    if not paper_path.exists():
        return False, "paper.md not found"
    text = paper_path.read_text()

    required = ["## Results", "## Method", "## Limitations"]
    missing = [r for r in required if r not in text]
    if missing:
        return False, f"paper.md missing sections: {missing}"

    repro = project_dir / "04_paper" / "reproducibility.md"
    if not repro.exists():
        return False, "reproducibility.md not found"
    return True, "ok"
