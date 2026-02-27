"""Pattern-driven ideation agent: uses knowledge graph patterns + history
to generate novel, evidence-grounded hypotheses."""

import json
from pathlib import Path

import yaml

from ..llm.router import get_router
from ..llm.prompts import IDEATION_ENHANCED_SYSTEM, IDEATION_ENHANCED_USER, IDEATION_SYSTEM, IDEATION_USER
from ..knowledge.kg_store import KnowledgeGraph
from ..knowledge.pattern_ranker import rank_patterns, format_patterns_for_prompt
from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def _load_taskspace(repo_root: Path) -> dict:
    ts_path = repo_root / "config" / "taskspace.yaml"
    return yaml.safe_load(ts_path.read_text())["taskspace"]


def _load_knowledge_config(repo_root: Path) -> dict:
    kc_path = repo_root / "config" / "knowledge.yaml"
    if kc_path.exists():
        return yaml.safe_load(kc_path.read_text()).get("knowledge", {})
    return {}


def run_ideation_enhanced(project_dir: Path, repo_root: Path) -> None:
    """Pattern-driven ideation with history awareness and fallback."""
    ts = _load_taskspace(repo_root)
    actions = ts["actions"]
    max_actions = ts["limits"]["max_actions_per_project"]
    action_ids = [a["id"] for a in actions]

    kc = _load_knowledge_config(repo_root)
    kg_path = repo_root / kc.get("kg_db_path", "artifacts/knowledge.db")

    has_patterns = False
    history_summary = "(no past experiments)"
    patterns_text = "(no patterns available - use free ideation)"

    if kg_path.exists():
        kg = KnowledgeGraph(kg_path)
        try:
            history_summary = kg.get_history_summary()

            ranked = rank_patterns(kg, repo_root)
            top_k = kc.get("patterns", {}).get("top_k_retrieval", 5)
            patterns_text = format_patterns_for_prompt(ranked, top_k=top_k)
            has_patterns = "(no " not in patterns_text
        finally:
            kg.close()

    actions_desc = "\n".join(
        "- %s: %s" % (a["id"], json.dumps(a["patch"])) for a in actions
    )

    if has_patterns:
        LOGGER.info("using pattern-enhanced ideation")
        user_prompt = IDEATION_ENHANCED_USER.format(
            taskspace_name=ts["name"],
            primary_metric=ts["baseline"]["primary_metric"],
            lower_is_better=not ts["baseline"].get("higher_is_better", True),
            history_summary=history_summary,
            top_patterns=patterns_text,
            actions_list=actions_desc,
            max_actions=max_actions,
        )
        system_prompt = IDEATION_ENHANCED_SYSTEM
    else:
        LOGGER.info("no patterns available, falling back to naive ideation with history")
        user_prompt = _build_naive_with_history_prompt(ts, actions_desc, max_actions, history_summary)
        system_prompt = IDEATION_SYSTEM

    router = get_router("azure_gpt4o")
    raw = router.generate(system_prompt, user_prompt, json_mode=True)

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
    pattern_id = result.get("pattern_id", None)
    rationale = result.get("rationale", "")
    novelty_note = result.get("novelty_note", "")

    idea = {
        "actions": chosen,
        "hypothesis": hypothesis,
        "patches": chosen_patches,
        "pattern_id": pattern_id,
        "rationale": rationale,
        "novelty_note": novelty_note,
    }

    idea_dir = project_dir / "00_idea"
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "idea.json").write_text(json.dumps(idea, indent=2))

    md_lines = [
        "# Idea",
        "",
        "**Hypothesis:** %s" % hypothesis,
        "",
        "**Actions:** %s" % ", ".join(chosen),
    ]
    if pattern_id:
        md_lines.append("")
        md_lines.append("**Based on pattern:** %s" % pattern_id)
    if rationale:
        md_lines.append("")
        md_lines.append("**Rationale:** %s" % rationale)
    if novelty_note:
        md_lines.append("")
        md_lines.append("**Novelty:** %s" % novelty_note)
    md_lines.append("")
    for aid in chosen:
        md_lines.append("- `%s`: %s" % (aid, json.dumps(chosen_patches[aid])))

    (idea_dir / "idea.md").write_text("\n".join(md_lines) + "\n")

    LOGGER.info(
        "ideation done (enhanced): actions=%s, pattern=%s",
        chosen, pattern_id or "none",
    )


def _build_naive_with_history_prompt(ts, actions_desc, max_actions, history_summary):
    """Naive ideation prompt but with history awareness to avoid repeats."""
    return (
        f"Task space: {ts['name']}\n"
        f"Baseline metric: {ts['baseline']['primary_metric']}\n\n"
        f"## Past experiments (DO NOT repeat these):\n{history_summary}\n\n"
        f"Available actions:\n{actions_desc}\n\n"
        f"Pick 1-{max_actions} actions and form a NOVEL hypothesis "
        f"(different from past experiments).\n"
        f'Output JSON:\n{{"actions": ["action_id", ...], "hypothesis": "one sentence hypothesis"}}\n'
    )
