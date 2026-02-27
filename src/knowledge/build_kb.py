"""One-command knowledge base builder.
Usage: python -m src.knowledge.build_kb [--collect] [--extract] [--mine] [--all]
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.knowledge.paper_collector import collect_papers
from src.knowledge.text_extractor import extract_all
from src.knowledge.method_extractor import extract_all as extract_method_units
from src.knowledge.kg_store import KnowledgeGraph
from src.knowledge.pattern_miner import mine_all_patterns
from src.knowledge.pattern_ranker import rank_patterns
from src.utils.log import get_logger

LOGGER = get_logger("build_kb")


def load_config(repo_root: Path) -> dict:
    kc_path = repo_root / "config" / "knowledge.yaml"
    if kc_path.exists():
        return yaml.safe_load(kc_path.read_text()).get("knowledge", {})
    return {}


def step_collect(repo_root: Path, cfg: dict) -> list[dict]:
    LOGGER.info("=== Step 1: Collect papers ===")
    coll = cfg.get("collection", {})
    keywords = coll.get("keywords", ["LLM finetuning"])
    max_papers = coll.get("max_papers", 200)
    pool_dir = repo_root / cfg.get("paper_pool_dir", "paper_pool")

    papers = collect_papers(keywords, max_papers=max_papers, pool_dir=pool_dir)
    LOGGER.info("collected %d papers", len(papers))
    return papers


def step_extract_text(repo_root: Path, cfg: dict, papers: list[dict]) -> dict:
    LOGGER.info("=== Step 2: Extract text ===")
    pool_dir = repo_root / cfg.get("paper_pool_dir", "paper_pool")
    save_dir = pool_dir / "papers"
    texts = extract_all(papers, save_dir=save_dir, rate_limit=0.5)
    LOGGER.info("extracted text for %d papers", len(texts))
    return texts


def step_extract_methods(cfg: dict, papers: list[dict], texts: dict, kg: KnowledgeGraph) -> int:
    LOGGER.info("=== Step 3: Extract method units ===")
    ext_cfg = cfg.get("extraction", {})
    max_units = ext_cfg.get("max_units_per_paper", 10)
    categories = ext_cfg.get("categories")

    units = extract_method_units(
        papers=papers,
        texts=texts,
        max_units_per_paper=max_units,
        categories=categories,
    )

    count = kg.add_method_units(units)
    LOGGER.info("added %d method units to KG", count)
    return count


def step_mine_patterns(cfg: dict, kg: KnowledgeGraph) -> int:
    LOGGER.info("=== Step 4: Mine patterns ===")
    pat_cfg = cfg.get("patterns", {})
    min_evidence = pat_cfg.get("min_evidence_count", 2)
    max_components = pat_cfg.get("max_components", 4)

    patterns = mine_all_patterns(kg, min_evidence=min_evidence, max_components=max_components)
    for p in patterns:
        kg.add_pattern(p)

    LOGGER.info("stored %d patterns in KG", len(patterns))
    return len(patterns)


def step_rank(repo_root: Path, kg: KnowledgeGraph) -> None:
    LOGGER.info("=== Step 5: Rank patterns ===")
    ranked = rank_patterns(kg, repo_root)
    LOGGER.info("ranked %d patterns", len(ranked))


def main():
    parser = argparse.ArgumentParser(description="Build FARS knowledge base")
    parser.add_argument("--collect", action="store_true", help="Collect papers")
    parser.add_argument("--extract", action="store_true", help="Extract text + method units")
    parser.add_argument("--mine", action="store_true", help="Mine and rank patterns")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if not any([args.collect, args.extract, args.mine, args.all]):
        args.all = True

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root)

    kg_path = repo_root / cfg.get("kg_db_path", "artifacts/knowledge.db")
    kg = KnowledgeGraph(kg_path)

    import json
    pool_dir = repo_root / cfg.get("paper_pool_dir", "paper_pool")
    metadata_path = pool_dir / "metadata.json"

    papers = []
    if args.collect or args.all:
        papers = step_collect(repo_root, cfg)

    if not papers and metadata_path.exists():
        papers = json.loads(metadata_path.read_text())

    if args.extract or args.all:
        if not papers:
            LOGGER.error("no papers to extract from, run --collect first")
        else:
            texts = step_extract_text(repo_root, cfg, papers)
            step_extract_methods(cfg, papers, texts, kg)

    if args.mine or args.all:
        step_mine_patterns(cfg, kg)
        step_rank(repo_root, kg)

    stats = kg.get_stats()
    LOGGER.info("KB stats: %s", stats)
    kg.close()


if __name__ == "__main__":
    main()
