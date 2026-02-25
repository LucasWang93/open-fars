"""Ideation agent: use GPT-4o to pick actions from taskspace and form hypothesis."""

import json
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

    actions_desc = chr(10).join(
        "- %s: %s" % (a["id"], json.dumps(a["patch"])) for a in actions
    )

    user_prompt = IDEATION_USER.format(
        taskspace_name=ts["name"],
        primary_metric=ts["baseline"]["primary_metric"],
        actions_list=actions_desc,
        max_actions=max_actions,
    )

    router = get_router("azure_gpt4o")
    raw = router.generate(IDEATION_SYSTEM, user_prompt, json_mode=True)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        LOGGER.error("LLM returned non-JSON, falling back: %s", raw[:200])
        result = {"actions": action_ids[:1], "hypothesis": "fallback hypothesis"}

    chosen = [a for a in result.get("actions", []) if a in action_ids]
    if not chosen:
        chosen = action_ids[:1]

    patches = {a["id"]: a["patch"] for a in actions}
    chosen_patches = {aid: patches[aid] for aid in chosen}
    hypothesis = result.get("hypothesis", "No hypothesis provided")

    idea = {
        "actions": chosen,
        "hypothesis": hypothesis,
        "patches": chosen_patches,
    }

    idea_dir = project_dir / "00_idea"
    idea_dir.mkdir(parents=True, exist_ok=True)

    (idea_dir / "idea.json").write_text(json.dumps(idea, indent=2))

    md = "# Idea\n\n**Hypothesis:** %s\n\n**Actions:** %s\n" % (
        hypothesis, ", ".join(chosen)
    )
    for aid in chosen:
        md += "\n- `%s`: %s\n" % (aid, json.dumps(chosen_patches[aid]))
    (idea_dir / "idea.md").write_text(md)

    LOGGER.info("ideation done (GPT-4o): actions=%s", chosen)
