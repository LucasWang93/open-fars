"""SQLite-backed project registry with row-level locking."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from ..utils.log import get_logger

LOGGER = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_dir TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'IDEA',
    retry_count INTEGER NOT NULL DEFAULT 0,
    failure_reason TEXT,
    locked_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    meta_json TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DDL)

    def register(self, project_id: str, project_dir: str, meta: dict) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO projects "
            "(project_id, project_dir, state, retry_count, created_at, updated_at, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                project_id,
                project_dir,
                meta["state"],
                meta.get("retry_count", 0),
                meta["created_at"],
                meta["updated_at"],
                json.dumps(meta),
            ),
        )
        self._conn.commit()

    def try_lock(self, project_id: str, owner: str = "daemon") -> bool:
        cur = self._conn.execute(
            "UPDATE projects SET locked_by = ? "
            "WHERE project_id = ? AND locked_by IS NULL",
            (owner, project_id),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def unlock(self, project_id: str) -> None:
        self._conn.execute(
            "UPDATE projects SET locked_by = NULL WHERE project_id = ?",
            (project_id,),
        )
        self._conn.commit()

    def update_state(self, project_id: str, state: str, meta: dict) -> None:
        self._conn.execute(
            "UPDATE projects SET state = ?, updated_at = ?, meta_json = ?, "
            "retry_count = ?, failure_reason = ? WHERE project_id = ?",
            (
                state,
                meta["updated_at"],
                json.dumps(meta),
                meta.get("retry_count", 0),
                meta.get("failure_reason"),
                project_id,
            ),
        )
        self._conn.commit()

    def get_project(self, project_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_by_state(self, state: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM projects WHERE state = ? AND locked_by IS NULL "
            "ORDER BY created_at",
            (state,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_active(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM projects WHERE state NOT IN ('DONE', 'ABORT') "
            "ORDER BY created_at",
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
