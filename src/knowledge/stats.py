"""Knowledge base and experiment statistics for monitoring."""

from pathlib import Path

import yaml

from ..utils.log import get_logger
from .kg_store import KnowledgeGraph

LOGGER = get_logger(__name__)


def compute_stats(repo_root: Path) -> dict:
    """Compute comprehensive stats from the knowledge graph."""
    kc_path = repo_root / "config" / "knowledge.yaml"
    if kc_path.exists():
        kc = yaml.safe_load(kc_path.read_text()).get("knowledge", {})
    else:
        kc = {}

    kg_path = repo_root / kc.get("kg_db_path", "artifacts/knowledge.db")
    if not kg_path.exists():
        return {"status": "no knowledge base found"}

    kg = KnowledgeGraph(kg_path)
    try:
        base = kg.get_stats()

        tried = kg.get_tried_combinations()
        total_exp = len(tried)
        success_count = sum(1 for t in tried if t["outcome"] == "success")
        negative_count = sum(1 for t in tried if t["outcome"] == "negative")
        failed_count = sum(1 for t in tried if t["outcome"] == "failed")

        action_sets = [frozenset(t["actions"]) for t in tried]
        unique_combos = len(set(action_sets))

        patterns = kg.get_all_patterns()
        patterns_with_mapping = sum(1 for p in patterns if p.mappable_actions)

        stats = {
            **base,
            "total_experiments": total_exp,
            "success_rate": success_count / max(total_exp, 1),
            "negative_rate": negative_count / max(total_exp, 1),
            "failed_rate": failed_count / max(total_exp, 1),
            "unique_action_combos": unique_combos,
            "novelty_rate": unique_combos / max(total_exp, 1),
            "patterns_with_action_mapping": patterns_with_mapping,
            "pattern_coverage": patterns_with_mapping / max(base.get("patterns", 1), 1),
        }
        return stats
    finally:
        kg.close()


def log_stats(repo_root: Path) -> None:
    """Log stats summary."""
    stats = compute_stats(repo_root)
    LOGGER.info("=== Knowledge Base Stats ===")
    LOGGER.info("  Method units: %d", stats.get("method_units", 0))
    LOGGER.info("  Patterns: %d (mapped: %d)",
                stats.get("patterns", 0), stats.get("patterns_with_action_mapping", 0))
    LOGGER.info("  Experiments: %d (success: %.0f%%, negative: %.0f%%, failed: %.0f%%)",
                stats.get("total_experiments", 0),
                stats.get("success_rate", 0) * 100,
                stats.get("negative_rate", 0) * 100,
                stats.get("failed_rate", 0) * 100)
    LOGGER.info("  Unique action combos: %d (novelty rate: %.0f%%)",
                stats.get("unique_action_combos", 0),
                stats.get("novelty_rate", 0) * 100)
    LOGGER.info("  Category breakdown: %s", stats.get("categories", {}))
