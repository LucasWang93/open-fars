"""Gate checks between pipeline stages."""

import json
from pathlib import Path
from typing import Tuple

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def gate_a_ideation(project_dir: Path, valid_action_ids: list = None) -> Tuple[bool, str]:
    """Validate idea.json output.

    For free-form ideation: checks hypothesis + method fields exist.
    For legacy action-based ideation: checks actions are valid.
    """
    idea_path = project_dir / "00_idea" / "idea.json"
    if not idea_path.exists():
        return False, "idea.json not found"
    try:
        idea = json.loads(idea_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"idea.json parse error: {exc}"

    if "hypothesis" not in idea or not idea["hypothesis"]:
        return False, "missing hypothesis"

    if "method" in idea and isinstance(idea["method"], dict):
        return True, "ok"

    if "actions" in idea:
        chosen = idea["actions"]
        if not chosen:
            return False, "no actions chosen"
        if valid_action_ids:
            for a in chosen:
                if a not in valid_action_ids:
                    return False, f"invalid action: {a}"
        return True, "ok"

    return False, "idea.json missing both 'method' and 'actions'"


def gate_b_experiment(project_dir: Path) -> Tuple[bool, str]:
    """Validate that experiment produced valid metrics.json files."""
    runs_dir = project_dir / "02_exp" / "runs"
    if not runs_dir.exists():
        return False, "runs directory missing"
    run_dirs = [d for d in sorted(runs_dir.iterdir()) if d.is_dir()]
    if not run_dirs:
        return False, "no run directories"

    success_count = 0
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
        if m.get("status") == "SUCCESS":
            if "primary_metric" not in m:
                return False, f"missing primary_metric in {rd.name}"
            success_count += 1

    if success_count == 0:
        return False, "all runs failed"

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
