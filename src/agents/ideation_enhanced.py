"""Free-form research ideation agent for GUI Agent action planning."""

import json
from pathlib import Path

import yaml

from ..llm.prompts import IDEATION_ENHANCED_SYSTEM, IDEATION_ENHANCED_USER
from ..llm.router import get_router
from ..utils.log import get_logger

LOGGER = get_logger(__name__)

REQUIRED_METHOD_KEYS = [
    "training_strategy", "data_processing", "prompt_design",
    "model_config", "augmentation", "key_innovation",
]


def _load_configs(repo_root: Path):
    ts = yaml.safe_load((repo_root / "config" / "taskspace.yaml").read_text())["taskspace"]
    kg_cfg = yaml.safe_load((repo_root / "config" / "knowledge.yaml").read_text())["knowledge"]
    return ts, kg_cfg


def _format_dimensions(ts: dict) -> str:
    lines = []
    for dim in ts.get("exploration_dimensions", []):
        examples = ", ".join(dim.get("examples", []))
        lines.append(f"- **{dim['name']}**: {dim.get('description', '')}  Examples: {examples}")
    return "\n".join(lines)


def _format_baseline(ts: dict) -> str:
    bl = ts.get("baseline", {})
    return yaml.dump(bl, default_flow_style=False)


def _get_history_and_patterns(repo_root: Path, kg_cfg: dict, ts: dict):
    history_summary = "No past experiments yet."
    top_patterns = "No patterns available."

    try:
        from ..knowledge.kg_store import KnowledgeGraph
        from ..knowledge.pattern_ranker import rank_patterns, format_patterns_for_prompt

        kg = KnowledgeGraph(repo_root / kg_cfg["kg_db_path"])
        history = kg.get_history_summary()
        if history:
            history_summary = history

        patterns = kg.get_all_patterns()
        if patterns:
            pcfg = kg_cfg.get("patterns", {})
            ranked = rank_patterns(patterns, ts, kg, top_k=pcfg.get("top_k_retrieval", 5))
            LOGGER.info("ranked %d patterns, top score=%.4f",
                        len(ranked), ranked[0][1] if ranked else 0)
            top_patterns = format_patterns_for_prompt(ranked[:5], kg)
    except Exception as exc:
        LOGGER.warning("knowledge graph unavailable: %s", exc)

    return history_summary, top_patterns


def run_ideation_enhanced(project_dir: Path, repo_root: Path) -> None:
    ts, kg_cfg = _load_configs(repo_root)

    history_summary, top_patterns = _get_history_and_patterns(repo_root, kg_cfg, ts)

    user_prompt = IDEATION_ENHANCED_USER.format(
        base_model=ts.get("base_model", "Qwen/Qwen3-4B-Instruct-2507"),
        primary_metric=ts.get("primary_metric", "step_success_rate"),
        higher_is_better=ts.get("higher_is_better", True),
        dimensions_text=_format_dimensions(ts),
        baseline_text=_format_baseline(ts),
        history_summary=history_summary,
        top_patterns=top_patterns,
    )

    LOGGER.info("using free-form enhanced ideation for GUI Agent research")
    router = get_router("azure_gpt4o")
    raw = router.generate(IDEATION_ENHANCED_SYSTEM, user_prompt, json_mode=True)
    idea = json.loads(raw)

    if "method" not in idea or not isinstance(idea["method"], dict):
        raise ValueError("LLM output missing 'method' dict")
    for key in REQUIRED_METHOD_KEYS:
        if key not in idea["method"]:
            idea["method"][key] = "default"

    if "hypothesis" not in idea or not idea["hypothesis"]:
        raise ValueError("LLM output missing 'hypothesis'")

    idea_dir = project_dir / "00_idea"
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "idea.json").write_text(json.dumps(idea, indent=2))

    md_lines = [
        f"# {idea.get('title', 'Untitled')}",
        "",
        f"**Hypothesis:** {idea['hypothesis']}",
        "",
        f"**Key Innovation:** {idea['method'].get('key_innovation', 'N/A')}",
        "",
        "## Method",
        f"- Training Strategy: {idea['method']['training_strategy']}",
        f"- Data Processing: {idea['method']['data_processing']}",
        f"- Prompt Design: {idea['method']['prompt_design']}",
        f"- Model Config: {idea['method']['model_config']}",
        f"- Augmentation: {idea['method']['augmentation']}",
        "",
        f"**Rationale:** {idea.get('rationale', 'N/A')}",
        f"**Novelty:** {idea.get('novelty_note', 'N/A')}",
    ]
    (idea_dir / "idea.md").write_text("\n".join(md_lines))

    LOGGER.info(
        "ideation done (enhanced): title=%s, strategy=%s, prompt=%s",
        idea.get("title", "?"),
        idea["method"]["training_strategy"],
        idea["method"]["prompt_design"],
    )
