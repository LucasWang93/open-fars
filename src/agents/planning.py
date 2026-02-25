"""Planning agent: use GPT-4o to generate experiment plan."""

import json
import copy
from pathlib import Path

import yaml

from ..llm.router import get_router
from ..llm.prompts import PLANNING_SYSTEM, PLANNING_USER
from ..utils.log import get_logger

LOGGER = get_logger(__name__)

BASELINE_CONFIG = {
    "train": {"learning_rate": 0.0001, "num_train_epochs": 1, "warmup_ratio": 0.03},
    "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
    "data": {"max_seq_length": 512, "batch_size": 4},
}


def _load_taskspace(repo_root: Path) -> dict:
    ts_path = repo_root / "config" / "taskspace.yaml"
    return yaml.safe_load(ts_path.read_text())["taskspace"]


def _apply_patches(base: dict, patches: dict) -> dict:
    cfg = copy.deepcopy(base)
    for key, val in patches.items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def run_planning(project_dir: Path, repo_root: Path) -> None:
    ts = _load_taskspace(repo_root)
    idea_path = project_dir / "00_idea" / "idea.json"
    idea = json.loads(idea_path.read_text())

    seeds = ts["limits"]["seeds"]

    merged_patches = {}
    for patch_dict in idea["patches"].values():
        merged_patches.update(patch_dict)

    treatment_config = _apply_patches(BASELINE_CONFIG, merged_patches)

    router = get_router("azure_gpt4o")
    user_prompt = PLANNING_USER.format(
        hypothesis=idea["hypothesis"],
        actions=", ".join(idea["actions"]),
        baseline_config=json.dumps(BASELINE_CONFIG, indent=2),
        seeds=seeds,
        primary_metric=ts["baseline"]["primary_metric"],
    )
    raw = router.generate(PLANNING_SYSTEM, user_prompt, json_mode=True)

    try:
        llm_plan = json.loads(raw)
    except json.JSONDecodeError:
        LOGGER.warning("LLM plan parse failed, using defaults")
        llm_plan = {}

    plan = {
        "plan_summary": llm_plan.get("plan_summary", "Test: " + idea["hypothesis"]),
        "variables": llm_plan.get("variables", list(merged_patches.keys())),
        "control": llm_plan.get("control", "baseline config"),
        "treatment": llm_plan.get("treatment", "apply " + ", ".join(idea["actions"])),
        "metric": ts["baseline"]["primary_metric"],
        "seeds": seeds,
        "models": ts.get("models", []),
        "budget_estimate_minutes": llm_plan.get("budget_estimate_minutes", len(seeds) * 10),
    }

    plan_dir = project_dir / "01_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2))

    configs = {
        "baseline": BASELINE_CONFIG,
        "treatment": treatment_config,
        "seeds": seeds,
        "models": ts.get("models", []),
    }
    with open(plan_dir / "config.yaml", "w") as f:
        yaml.dump(configs, f, default_flow_style=False)

    md_lines = [
        "# Experiment Plan",
        "",
        "**Hypothesis:** " + idea["hypothesis"],
        "",
        "## Design",
        "- **Control:** baseline config",
        "- **Treatment:** " + plan["treatment"],
        "- **Variables:** " + ", ".join(plan["variables"]),
        "- **Metric:** " + plan["metric"],
        "- **Seeds:** " + str(seeds),
        "- **Models:** " + ", ".join(plan.get("models", [])),
        "- **Budget:** ~%d minutes" % plan["budget_estimate_minutes"],
        "",
        "## Configs",
        "```yaml",
        yaml.dump(configs, default_flow_style=False).strip(),
        "```",
    ]
    (plan_dir / "plan.md").write_text("\n".join(md_lines) + "\n")

    LOGGER.info("planning done (GPT-4o): %d seeds, variables=%s", len(seeds), plan["variables"])
