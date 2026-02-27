"""Mine research patterns from the knowledge graph via co-occurrence and graph traversal."""

import hashlib
from collections import Counter, defaultdict
from itertools import combinations

from ..utils.log import get_logger
from .kg_store import KnowledgeGraph
from .schemas import MethodUnit, MethodRelation, ResearchPattern

LOGGER = get_logger(__name__)


def _pattern_id(names: list[str]) -> str:
    key = "::".join(sorted(names))
    return "pat_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def mine_cooccurrence_patterns(
    kg: KnowledgeGraph,
    min_evidence: int = 2,
    max_components: int = 4,
) -> list[ResearchPattern]:
    """Find method unit combinations that co-occur in the same paper."""
    units = kg.query_all_units()

    paper_to_units: dict[str, list[MethodUnit]] = defaultdict(list)
    for u in units:
        if u.paper_source:
            paper_to_units[u.paper_source].append(u)

    combo_counter: Counter[tuple[str, ...]] = Counter()
    combo_details: dict[tuple[str, ...], dict] = {}

    for paper_id, paper_units in paper_to_units.items():
        if len(paper_units) < 2:
            continue

        for size in range(2, min(max_components + 1, len(paper_units) + 1)):
            for combo in combinations(paper_units, size):
                names = tuple(sorted(u.name for u in combo))
                combo_counter[names] += 1

                if names not in combo_details:
                    combo_details[names] = {
                        "unit_ids": [u.unit_id for u in combo],
                        "names": list(names),
                        "categories": list({u.category for u in combo}),
                        "papers": set(),
                        "outputs": [],
                    }
                combo_details[names]["papers"].add(paper_id)
                for u in combo:
                    combo_details[names]["outputs"].extend(u.outputs)

    patterns = []
    for names, count in combo_counter.most_common():
        if count < min_evidence:
            break

        details = combo_details[names]
        outputs = list(set(details["outputs"]))
        benefit = ", ".join(outputs[:3]) if outputs else "combined effect"

        pattern = ResearchPattern(
            pattern_id=_pattern_id(list(names)),
            components=details["unit_ids"],
            component_names=details["names"],
            expected_benefit=benefit,
            evidence_count=count,
            source_papers=list(details["papers"]),
        )
        patterns.append(pattern)

    LOGGER.info("mined %d co-occurrence patterns (min_evidence=%d)", len(patterns), min_evidence)
    return patterns


def mine_neighbor_patterns(
    kg: KnowledgeGraph,
    max_components: int = 3,
) -> list[ResearchPattern]:
    """Create patterns from KG neighbors (related method units)."""
    units = kg.query_all_units()
    patterns = []
    seen = set()

    for unit in units:
        neighbors = kg.query_neighbors(unit.unit_id)
        if not neighbors:
            continue

        for neighbor_id, rel_type, weight in neighbors:
            neighbor = kg.get_method_unit(neighbor_id)
            if not neighbor:
                continue

            names = tuple(sorted([unit.name, neighbor.name]))
            if names in seen:
                continue
            seen.add(names)

            benefit_parts = list(set(unit.outputs + neighbor.outputs))
            benefit = ", ".join(benefit_parts[:3]) if benefit_parts else rel_type

            pattern = ResearchPattern(
                pattern_id=_pattern_id(list(names)),
                components=[unit.unit_id, neighbor_id],
                component_names=list(names),
                expected_benefit=benefit,
                evidence_count=1,
                source_papers=list({unit.paper_source, neighbor.paper_source} - {""}),
            )
            patterns.append(pattern)

    LOGGER.info("mined %d neighbor patterns", len(patterns))
    return patterns


def build_relations_from_cooccurrence(kg: KnowledgeGraph) -> int:
    """Add co_occurs_with relations for units from the same paper."""
    units = kg.query_all_units()
    paper_to_units: dict[str, list[MethodUnit]] = defaultdict(list)
    for u in units:
        if u.paper_source:
            paper_to_units[u.paper_source].append(u)

    count = 0
    for paper_id, paper_units in paper_to_units.items():
        for a, b in combinations(paper_units, 2):
            rel = MethodRelation(
                from_id=a.unit_id,
                to_id=b.unit_id,
                relation_type="co_occurs_with",
                weight=1.0,
                paper_source=paper_id,
            )
            kg.add_relation(rel)
            count += 1

    LOGGER.info("added %d co-occurrence relations", count)
    return count


def mine_all_patterns(
    kg: KnowledgeGraph,
    min_evidence: int = 2,
    max_components: int = 4,
) -> list[ResearchPattern]:
    """Run all pattern mining strategies and merge results."""
    build_relations_from_cooccurrence(kg)

    cooc = mine_cooccurrence_patterns(kg, min_evidence, max_components)
    neighbor = mine_neighbor_patterns(kg, max_components)

    seen_ids = set()
    merged = []
    for p in cooc + neighbor:
        if p.pattern_id not in seen_ids:
            seen_ids.add(p.pattern_id)
            merged.append(p)

    LOGGER.info("total unique patterns after merge: %d", len(merged))
    return merged
