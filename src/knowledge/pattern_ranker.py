"""Rank and score research patterns for ideation relevance."""

import json
from pathlib import Path
from typing import Optional

import yaml

from ..utils.log import get_logger
from .kg_store import KnowledgeGraph
from .schemas import ResearchPattern

LOGGER = get_logger(__name__)

ACTION_KEYWORD_MAP = {
    "lr_up": ["learning rate", "lr", "higher learning rate", "large lr"],
    "lr_down": ["learning rate", "lr", "lower learning rate", "small lr", "lr decay"],
    "warmup_more": ["warmup", "warm-up", "warmup ratio", "warmup steps"],
    "epochs_up": ["epoch", "training duration", "longer training", "more epochs"],
    "lora_rank_up": ["lora", "rank", "higher rank", "lora rank"],
    "lora_rank_down": ["lora", "rank", "lower rank", "lora rank", "compression"],
}


def _compute_action_mapping(pattern: ResearchPattern, taskspace_actions: list[dict]) -> list[str]:
    """Map pattern components to taskspace action IDs based on keyword overlap."""
    action_ids = []
    pattern_text = " ".join(pattern.component_names).lower() + " " + pattern.expected_benefit.lower()

    for action in taskspace_actions:
        aid = action["id"]
        keywords = ACTION_KEYWORD_MAP.get(aid, [aid.replace("_", " ")])
        for kw in keywords:
            if kw.lower() in pattern_text:
                if aid not in action_ids:
                    action_ids.append(aid)
                break

    return action_ids


def score_pattern(
    pattern: ResearchPattern,
    taskspace_actions: list[dict],
    tried_action_sets: list[set[str]],
) -> float:
    """Score a pattern based on evidence, feasibility, and novelty."""
    mappable = _compute_action_mapping(pattern, taskspace_actions)
    pattern.mappable_actions = mappable

    if not mappable:
        return 0.0

    evidence_score = min(pattern.evidence_count / 5.0, 1.0)
    feasibility_score = len(mappable) / max(len(pattern.component_names), 1)

    action_set = frozenset(mappable)
    novelty_score = 1.0
    for tried in tried_action_sets:
        if action_set == tried:
            novelty_score = 0.0
            break
        overlap = len(action_set & tried) / max(len(action_set | tried), 1)
        novelty_score = min(novelty_score, 1.0 - overlap)

    score = (
        0.3 * evidence_score
        + 0.3 * feasibility_score
        + 0.4 * novelty_score
    )

    return round(score, 4)


def rank_patterns(
    kg: KnowledgeGraph,
    repo_root: Path,
) -> list[ResearchPattern]:
    """Load patterns, score them against taskspace + history, return ranked list."""
    ts_path = repo_root / "config" / "taskspace.yaml"
    ts = yaml.safe_load(ts_path.read_text())["taskspace"]
    taskspace_actions = ts["actions"]

    tried = kg.get_tried_combinations()
    tried_action_sets = [frozenset(t["actions"]) for t in tried]

    patterns = kg.get_all_patterns()
    for p in patterns:
        p.quality_score = score_pattern(p, taskspace_actions, tried_action_sets)
        kg.add_pattern(p)

    patterns.sort(key=lambda p: p.quality_score, reverse=True)
    LOGGER.info("ranked %d patterns, top score=%.4f",
                len(patterns), patterns[0].quality_score if patterns else 0)
    return patterns


def format_patterns_for_prompt(patterns: list[ResearchPattern], top_k: int = 5) -> str:
    """Format top patterns as human-readable text for LLM prompts."""
    if not patterns:
        return "(no patterns available - use free ideation)"

    lines = []
    for p in patterns[:top_k]:
        if p.quality_score <= 0:
            continue
        actions_str = ", ".join(p.mappable_actions) if p.mappable_actions else "none mapped"
        lines.append(
            f"- Pattern {p.pattern_id}: [{', '.join(p.component_names)}] "
            f"-> {p.expected_benefit} "
            f"(evidence: {p.evidence_count} papers, mappable actions: [{actions_str}], "
            f"score: {p.quality_score:.2f})"
        )

    if not lines:
        return "(no high-quality patterns found - use free ideation)"

    return "\n".join(lines)
