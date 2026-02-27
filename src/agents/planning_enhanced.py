"""Pattern-guided planning agent: uses pattern structure to generate
more grounded experiment plans with literature references."""

import json
import copy
from pathlib import Path

import yaml

from ..llm.router import get_router
from ..llm.prompts import PLANNING_SYSTEM, PLANNING_USER
from ..knowledge.kg_store import KnowledgeGraph
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


def _load_knowledge_config(repo_root: Path) -> dict:
    kc_path = repo_root / "config" / "knowledge.yaml"
    if kc_path.exists():
        return yaml.safe_load(kc_path.read_text()).get("knowledge", {})
    return {}


def _apply_patches(base: dict, patches: dict) -> dict:
    cfg = copy.deepcopy(base)
    for key, val in patches.items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def _get_pattern_context(idea: dict, repo_root: Path) -> str:
    pattern_id = idea.get("pattern_id")
    if not pattern_id:
        return ""

    kc = _load_knowledge_config(repo_root)
    kg_path = repo_root / kc.get("kg_db_path", "artifacts/knowledge.db")
    if not kg_path.exists():
        return ""

    kg = KnowledgeGraph(kg_path)
    try:
        patterns = kg.get_all_patterns()
        for p in patterns:
            if p.pattern_id == pattern_id:
                return (
                    "\nPattern context (%s):\n"
                    "- Components: %s\n"
                    "- Expected benefit: %s\n"
                    "- Evidence: %d papers\n"
                    "- Source papers: %s\n"
                    % (p.pattern_id, ", ".join(p.component_names),
                       p.expected_benefit, p.evidence_count,
                       ", ".join(p.source_papers[:3]))
                )
    finally:
        kg.close()

    return ""


def run_planning_enhanced(project_dir: Path, repo_root: Path) -> None:
    ts = _load_taskspace(repo_root)
    idea_path = project_dir / "00_idea" / "idea.json"
    idea = json.loads(idea_path.read_text())

    seeds = ts["limits"]["seeds"]

    merged_patches = {}
    for patch_dict in idea["patches"].values():
        merged_patches.update(patch_dict)

    treatment_config = _apply_patches(BASELINE_CONFIG, merged_patches)

    pattern_context = _get_pattern_context(idea, repo_root)
    rationale = idea.get("rationale", "")

    router = get_router("azure_gpt4o")
    enhanced_hypothesis = idea["hypothesis"]
    if rationale:
        enhanced_hypothesis += " (Rationale: %s)" % rationale

    user_prompt = PLANNING_USER.format(
        hypothesis=enhanced_hypothesis + pattern_context,
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
        "pattern_id": idea.get("pattern_id"),
        "rationale": rationale,
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
    ]
    if idea.get("pattern_id"):
        md_lines.append("")
        md_lines.append("**Based on pattern:** " + idea["pattern_id"])
    if rationale:
        md_lines.append("")
        md_lines.append("**Rationale:** " + rationale)
    md_lines.extend([
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
    ])
    (plan_dir / "plan.md").write_text("\n".join(md_lines) + "\n")

    LOGGER.info(
        "planning done (enhanced): %d seeds, variables=%s, pattern=%s",
        len(seeds), plan["variables"], plan.get("pattern_id") or "none",
    )
