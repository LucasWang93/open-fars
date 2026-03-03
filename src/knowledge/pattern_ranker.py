"""Rank and score research patterns for GUI Agent ideation."""

from pathlib import Path
from typing import Optional

import yaml

from ..utils.log import get_logger
from .kg_store import KnowledgeGraph
from .schemas import ResearchPattern

LOGGER = get_logger(__name__)

GUI_AGENT_KEYWORDS = [
    "action", "grounding", "element", "click", "navigation", "web",
    "gui", "screen", "agent", "planning", "task", "prompt", "chain",
    "thought", "curriculum", "augmentation", "training", "fine-tune",
    "sft", "instruction", "benchmark", "mind2web", "webarena",
]


def _relevance_score(pattern: ResearchPattern) -> float:
    """Score how relevant a pattern is to GUI Agent research."""
    text = " ".join(pattern.component_names).lower()
    if hasattr(pattern, "expected_benefit"):
        text += " " + pattern.expected_benefit.lower()

    hits = sum(1 for kw in GUI_AGENT_KEYWORDS if kw in text)
    return min(hits / 5.0, 1.0)


def score_pattern(
    pattern: ResearchPattern,
    tried_descriptions: list[str] = None,
) -> float:
    """Score a pattern based on evidence, relevance, and novelty."""
    evidence_score = min(pattern.evidence_count / 5.0, 1.0)
    relevance_score = _relevance_score(pattern)

    novelty_score = 1.0
    if tried_descriptions:
        pattern_text = " ".join(pattern.component_names).lower()
        for desc in tried_descriptions:
            desc_lower = desc.lower()
            overlap_words = set(pattern_text.split()) & set(desc_lower.split())
            overlap_ratio = len(overlap_words) / max(len(set(pattern_text.split())), 1)
            novelty_score = min(novelty_score, 1.0 - overlap_ratio)

    score = (
        0.3 * evidence_score
        + 0.4 * relevance_score
        + 0.3 * novelty_score
    )
    return round(score, 4)


def rank_patterns(
    patterns: list,
    ts: dict,
    kg: KnowledgeGraph,
    top_k: int = 5,
) -> list[tuple]:
    """Score and rank patterns. Returns list of (pattern, score) tuples."""
    tried = kg.get_tried_combinations()
    tried_descriptions = []
    for t in tried:
        actions = t.get("actions", [])
        hypothesis = t.get("hypothesis", "")
        tried_descriptions.append(" ".join(actions) + " " + hypothesis)

    scored = []
    for p in patterns:
        s = score_pattern(p, tried_descriptions)
        scored.append((p, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    LOGGER.info("ranked %d patterns, top score=%.4f",
                len(scored), scored[0][1] if scored else 0)
    return scored[:top_k * 10]


def format_patterns_for_prompt(
    scored_patterns: list[tuple],
    kg: KnowledgeGraph,
    top_k: int = 5,
) -> str:
    """Format top patterns as text for LLM prompts."""
    if not scored_patterns:
        return "(no patterns available)"

    lines = []
    for p, score in scored_patterns[:top_k]:
        if score <= 0:
            continue
        components = ", ".join(p.component_names) if hasattr(p, "component_names") else str(p.pattern_id)
        benefit = p.expected_benefit if hasattr(p, "expected_benefit") else "N/A"
        lines.append(
            f"- Pattern {p.pattern_id}: [{components}] "
            f"-> {benefit} "
            f"(evidence: {p.evidence_count} papers, score: {score:.2f})"
        )

    if not lines:
        return "(no relevant patterns found)"

    return "\n".join(lines)
