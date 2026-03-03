"""Planning agent: convert free-form research idea into executable experiment config."""

import json
from pathlib import Path

import yaml

from ..llm.prompts import PLANNING_SYSTEM, PLANNING_USER
from ..llm.router import get_router
from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def run_planning_enhanced(project_dir: Path, repo_root: Path) -> None:
    ts = yaml.safe_load(
        (repo_root / "config" / "taskspace.yaml").read_text()
    )["taskspace"]

    idea = json.loads((project_dir / "00_idea" / "idea.json").read_text())

    user_prompt = PLANNING_USER.format(
        title=idea.get("title", "Untitled"),
        hypothesis=idea["hypothesis"],
        method_json=json.dumps(idea["method"], indent=2),
        config_hints_json=json.dumps(idea.get("config_hints", {}), indent=2),
        baseline_config=yaml.dump(ts.get("baseline", {}), default_flow_style=False),
        base_model=ts.get("base_model", "Qwen/Qwen3-4B-Instruct-2507"),
        seeds=ts.get("constraints", {}).get("seeds", [42, 123]),
        max_gpu_hours=ts.get("constraints", {}).get("max_gpu_hours", 6),
        max_train_samples=ts.get("baseline", {}).get("data", {}).get("max_train_samples", 2000),
        max_eval_samples=ts.get("baseline", {}).get("data", {}).get("max_eval_samples", 500),
        primary_metric=ts.get("primary_metric", "step_success_rate"),
    )

    router = get_router("azure_gpt4o")
    raw = router.generate(PLANNING_SYSTEM, user_prompt, json_mode=True)
    plan = json.loads(raw)

    plan_dir = project_dir / "01_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "base_model": ts.get("base_model", "Qwen/Qwen3-4B-Instruct-2507"),
        "benchmark": ts.get("benchmark", "mind2web"),
        "seeds": ts.get("constraints", {}).get("seeds", [42, 123]),
        "primary_metric": ts.get("primary_metric", "step_success_rate"),
        "evaluation_metrics": ts.get("evaluation_metrics", []),
        "baseline": plan.get("baseline", ts.get("baseline", {})),
        "treatment": plan.get("treatment", {}),
    }
    (plan_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2))

    md_lines = [
        "# Experiment Plan",
        "",
        f"**Title:** {idea.get('title', 'Untitled')}",
        f"**Hypothesis:** {idea['hypothesis']}",
        "",
        f"**Based on pattern:** {idea.get('pattern_id', 'N/A')}",
        "",
        f"## Summary",
        plan.get("plan_summary", "N/A"),
        "",
        "## Design",
        f"- **Variables:** {', '.join(plan.get('variables', []))}",
        f"- **Metric:** {plan.get('metric', ts.get('primary_metric', 'step_success_rate'))}",
        f"- **Seeds:** {config['seeds']}",
        f"- **Model:** {config['base_model']}",
        f"- **Budget:** ~{plan.get('budget_estimate_minutes', 120)} minutes",
        "",
        "## Baseline",
        "```yaml",
        yaml.dump(config["baseline"], default_flow_style=False).strip(),
        "```",
        "",
        "## Treatment",
        "```yaml",
        yaml.dump(config["treatment"], default_flow_style=False).strip(),
        "```",
    ]
    (plan_dir / "plan.md").write_text("\n".join(md_lines))

    LOGGER.info(
        "planning done (enhanced): variables=%s, treatment_strategy=%s",
        plan.get("variables", []),
        config["treatment"].get("training_strategy", "?"),
    )
