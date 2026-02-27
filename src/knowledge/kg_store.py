"""SQLite-backed knowledge graph for method units, relations, patterns, and experiment history."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from ..utils.log import get_logger
from .schemas import MethodUnit, MethodRelation, ResearchPattern, ExperimentRecord

LOGGER = get_logger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS method_units (
    unit_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    inputs TEXT,        -- JSON array
    outputs TEXT,       -- JSON array
    paper_source TEXT,
    paper_title TEXT,
    confidence REAL DEFAULT 0.8
);

CREATE TABLE IF NOT EXISTS method_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    paper_source TEXT,
    UNIQUE(from_id, to_id, relation_type)
);

CREATE TABLE IF NOT EXISTS research_patterns (
    pattern_id TEXT PRIMARY KEY,
    components TEXT,         -- JSON array of unit_ids
    component_names TEXT,    -- JSON array of names
    expected_benefit TEXT,
    evidence_count INTEGER DEFAULT 0,
    quality_score REAL DEFAULT 0.0,
    mappable_actions TEXT,   -- JSON array
    source_papers TEXT       -- JSON array
);

CREATE TABLE IF NOT EXISTS experiment_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    pattern_id TEXT,
    actions TEXT,            -- JSON array
    hypothesis TEXT,
    outcome TEXT,
    eval_loss REAL,
    timestamp TEXT,
    UNIQUE(project_id)
);

CREATE INDEX IF NOT EXISTS idx_mu_category ON method_units(category);
CREATE INDEX IF NOT EXISTS idx_mr_from ON method_relations(from_id);
CREATE INDEX IF NOT EXISTS idx_mr_to ON method_relations(to_id);
CREATE INDEX IF NOT EXISTS idx_eh_actions ON experiment_history(actions);
"""


class KnowledgeGraph:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self):
        self._conn.close()

    # -- Method Units --

    def add_method_unit(self, unit: MethodUnit) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO method_units
               (unit_id, name, category, description, inputs, outputs,
                paper_source, paper_title, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (unit.unit_id, unit.name, unit.category, unit.description,
             json.dumps(unit.inputs), json.dumps(unit.outputs),
             unit.paper_source, unit.paper_title, unit.confidence),
        )
        self._conn.commit()

    def add_method_units(self, units: list[MethodUnit]) -> int:
        count = 0
        for u in units:
            try:
                self.add_method_unit(u)
                count += 1
            except Exception as exc:
                LOGGER.warning("failed to add unit %s: %s", u.unit_id, exc)
        LOGGER.info("added %d/%d method units to KG", count, len(units))
        return count

    def get_method_unit(self, unit_id: str) -> Optional[MethodUnit]:
        row = self._conn.execute(
            "SELECT * FROM method_units WHERE unit_id = ?", (unit_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_unit(row)

    def query_by_category(self, category: str) -> list[MethodUnit]:
        rows = self._conn.execute(
            "SELECT * FROM method_units WHERE category = ? ORDER BY confidence DESC",
            (category,),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def query_all_units(self) -> list[MethodUnit]:
        rows = self._conn.execute(
            "SELECT * FROM method_units ORDER BY confidence DESC"
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def search_units(self, keyword: str) -> list[MethodUnit]:
        rows = self._conn.execute(
            "SELECT * FROM method_units WHERE name LIKE ? OR description LIKE ?",
            (f"%{keyword}%", f"%{keyword}%"),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def count_units(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM method_units").fetchone()[0]

    # -- Relations --

    def add_relation(self, rel: MethodRelation) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO method_relations
               (from_id, to_id, relation_type, weight, paper_source)
               VALUES (?, ?, ?, ?, ?)""",
            (rel.from_id, rel.to_id, rel.relation_type, rel.weight, rel.paper_source),
        )
        self._conn.commit()

    def query_neighbors(self, unit_id: str) -> list[tuple[str, str, float]]:
        """Return [(neighbor_id, relation_type, weight)] for a unit."""
        rows = self._conn.execute(
            """SELECT to_id, relation_type, weight FROM method_relations WHERE from_id = ?
               UNION
               SELECT from_id, relation_type, weight FROM method_relations WHERE to_id = ?""",
            (unit_id, unit_id),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    # -- Patterns --

    def add_pattern(self, pattern: ResearchPattern) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO research_patterns
               (pattern_id, components, component_names, expected_benefit,
                evidence_count, quality_score, mappable_actions, source_papers)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (pattern.pattern_id,
             json.dumps(pattern.components),
             json.dumps(pattern.component_names),
             pattern.expected_benefit,
             pattern.evidence_count,
             pattern.quality_score,
             json.dumps(pattern.mappable_actions),
             json.dumps(pattern.source_papers)),
        )
        self._conn.commit()

    def get_all_patterns(self) -> list[ResearchPattern]:
        rows = self._conn.execute(
            "SELECT * FROM research_patterns ORDER BY quality_score DESC"
        ).fetchall()
        return [self._row_to_pattern(r) for r in rows]

    def get_top_patterns(self, k: int = 5) -> list[ResearchPattern]:
        rows = self._conn.execute(
            "SELECT * FROM research_patterns ORDER BY quality_score DESC LIMIT ?",
            (k,),
        ).fetchall()
        return [self._row_to_pattern(r) for r in rows]

    def count_patterns(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM research_patterns").fetchone()[0]

    # -- Experiment History --

    def record_experiment(self, record: ExperimentRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO experiment_history
               (project_id, pattern_id, actions, hypothesis, outcome, eval_loss, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (record.project_id, record.pattern_id,
             json.dumps(record.actions), record.hypothesis,
             record.outcome, record.eval_loss, record.timestamp),
        )
        self._conn.commit()

    def get_tried_combinations(self) -> list[dict]:
        """Return list of {actions, hypothesis, outcome} for all past experiments."""
        rows = self._conn.execute(
            "SELECT actions, hypothesis, outcome, eval_loss FROM experiment_history ORDER BY id"
        ).fetchall()
        results = []
        for r in rows:
            results.append({
                "actions": json.loads(r[0]) if r[0] else [],
                "hypothesis": r[1] or "",
                "outcome": r[2] or "",
                "eval_loss": r[3],
            })
        return results

    def get_history_summary(self) -> str:
        """Human-readable summary of past experiments for prompts."""
        tried = self.get_tried_combinations()
        if not tried:
            return "(no past experiments)"
        lines = []
        for i, t in enumerate(tried, 1):
            acts = ", ".join(t["actions"])
            loss = t["eval_loss"]
            loss_str = f"eval_loss={loss:.4f}" if loss is not None else "no result"
            lines.append(f"{i}. actions=[{acts}] -> {t['outcome']} ({loss_str})")
        return "\n".join(lines)

    def count_experiments(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM experiment_history").fetchone()[0]

    # -- Stats --

    def get_stats(self) -> dict:
        return {
            "method_units": self.count_units(),
            "patterns": self.count_patterns(),
            "experiments": self.count_experiments(),
            "categories": self._get_category_counts(),
        }

    def _get_category_counts(self) -> dict:
        rows = self._conn.execute(
            "SELECT category, COUNT(*) FROM method_units GROUP BY category"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # -- Helpers --

    @staticmethod
    def _row_to_unit(row) -> MethodUnit:
        return MethodUnit(
            unit_id=row["unit_id"],
            name=row["name"],
            category=row["category"],
            description=row["description"] or "",
            inputs=json.loads(row["inputs"]) if row["inputs"] else [],
            outputs=json.loads(row["outputs"]) if row["outputs"] else [],
            paper_source=row["paper_source"] or "",
            paper_title=row["paper_title"] or "",
            confidence=row["confidence"] or 0.8,
        )

    @staticmethod
    def _row_to_pattern(row) -> ResearchPattern:
        return ResearchPattern(
            pattern_id=row["pattern_id"],
            components=json.loads(row["components"]) if row["components"] else [],
            component_names=json.loads(row["component_names"]) if row["component_names"] else [],
            expected_benefit=row["expected_benefit"] or "",
            evidence_count=row["evidence_count"] or 0,
            quality_score=row["quality_score"] or 0.0,
            mappable_actions=json.loads(row["mappable_actions"]) if row["mappable_actions"] else [],
            source_papers=json.loads(row["source_papers"]) if row["source_papers"] else [],
        )
