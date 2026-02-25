"""Ideation agent: pick actions from taskspace, write idea.json + idea.md."""

import json
import random
from pathlib import Path

import yaml

from ..llm.router import get_router
from ..llm.prompts import IDEATION_SYSTEM, IDEATION_USER
from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def _load_taskspace(repo_root: Path) -> dict:
    ts_path = repo_root / "config" / "taskspace.yaml"
    return yaml.safe_load(ts_path.read_text())["taskspace"]


def run_ideation(project_dir: Path, repo_root: Path) -> None:
    ts = _load_taskspace(repo_root)
    actions = ts["actions"]
    max_actions = ts["limits"]["max_actions_per_project"]
    action_ids = [a["id"] for a in actions]

    router = get_router("mock")

    k = random.randint(1, min(max_actions, len(actions)))
    chosen = random.sample(action_ids, k)
    patches = {a["id"]: a["patch"] for a in actions}
    chosen_patches = {aid: patches[aid] for aid in chosen}

    hypothesis = (
        f"Applying {', '.join(chosen)} to the baseline will improve "
        f"{ts['baseline']['primary_metric']}."
    )

    idea = {
        "actions": chosen,
        "hypothesis": hypothesis,
        "patches": chosen_patches,
    }

    idea_dir = project_dir / "00_idea"
    idea_dir.mkdir(parents=True, exist_ok=True)

    (idea_dir / "idea.json").write_text(json.dumps(idea, indent=2))

    md = f"# Idea\n\n**Hypothesis:** {hypothesis}\n\n**Actions:** {', '.join(chosen)}\n"
    for aid in chosen:
        md += f"\n- `{aid}`: {json.dumps(chosen_patches[aid])}\n"
    (idea_dir / "idea.md").write_text(md)

    LOGGER.info("ideation done: actions=%s", chosen)
