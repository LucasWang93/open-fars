"""Planning agent: generate plan.md + config.yaml from idea."""

import json
import copy
from pathlib import Path

import yaml

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def _load_taskspace(repo_root: Path) -> dict:
    ts_path = repo_root / "config" / "taskspace.yaml"
    return yaml.safe_load(ts_path.read_text())["taskspace"]


def _apply_patches(base: dict, patches: dict) -> dict:
    """Apply dot-notation patches to a flat config dict."""
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
    baseline_config = {
        "train": {"lr": 0.0001, "epochs": 5},
        "gen": {"temperature": 1.0},
    }

    merged_patches = {}
    for patch_dict in idea["patches"].values():
        merged_patches.update(patch_dict)

    treatment_config = _apply_patches(baseline_config, merged_patches)

    plan = {
        "plan_summary": f"Test hypothesis: {idea['hypothesis']}",
        "variables": list(merged_patches.keys()),
        "control": "baseline config (no changes)",
        "treatment": f"apply {', '.join(idea['actions'])}",
        "metric": ts["baseline"]["primary_metric"],
        "seeds": seeds,
        "budget_estimate_minutes": len(seeds) * 2,
    }

    plan_dir = project_dir / "01_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2))

    configs = {
        "baseline": baseline_config,
        "treatment": treatment_config,
        "seeds": seeds,
    }
    with open(plan_dir / "config.yaml", "w") as f:
        yaml.dump(configs, f, default_flow_style=False)

    md_lines = [
        "# Experiment Plan",
        "",
        f"**Hypothesis:** {idea['hypothesis']}",
        "",
        "## Design",
        f"- **Control:** baseline config",
        f"- **Treatment:** {plan['treatment']}",
        f"- **Variables:** {', '.join(plan['variables'])}",
        f"- **Metric:** {plan['metric']}",
        f"- **Seeds:** {seeds}",
        f"- **Budget:** ~{plan['budget_estimate_minutes']} minutes",
        "",
        "## Configs",
        "```yaml",
        yaml.dump(configs, default_flow_style=False).strip(),
        "```",
    ]
    (plan_dir / "plan.md").write_text("\n".join(md_lines) + "\n")

    LOGGER.info("planning done: %d seeds, variables=%s", len(seeds), plan["variables"])
