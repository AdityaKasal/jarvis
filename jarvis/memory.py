"""Conversation memory: SQLite, and it outlives the process.

Three layers, cheapest first:

  * `turns`  - every line either side has said, verbatim.
  * `sessions.summary` - once a session outgrows the prompt budget, its older
    turns are folded into prose and dropped from the context.
  * `facts` - durable things about the user, written deliberately by Claude via
    the remember_fact tool. These survive every session and are always in the
    prompt, which is why they have to be small and few.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    summary     TEXT,
    -- Highest turn id already folded into `summary`. Turns above it are still
    -- replayed verbatim; turns at or below it exist only as the summary.
    summary_through INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns(session_id, id);
CREATE TABLE IF NOT EXISTS facts (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


class Memory:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- sessions ---

    def start_session(self) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO sessions (started_at) VALUES (?)",
                (datetime.now().isoformat(timespec="seconds"),),
            )
            return int(cur.lastrowid)

    def end_session(self, session_id: int, summary: Optional[str] = None) -> None:
        with self._conn() as c:
            if summary:
                c.execute(
                    "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
                    (datetime.now().isoformat(timespec="seconds"), summary, session_id),
                )
            else:
                c.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ?",
                    (datetime.now().isoformat(timespec="seconds"), session_id),
                )

    def session_summary(self, session_id: int) -> tuple[Optional[str], int]:
        """This session's rolling summary and the turn id it covers up to."""
        with self._conn() as c:
            row = c.execute(
                "SELECT summary, summary_through FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None, 0
        return row["summary"], row["summary_through"]

    def set_session_summary(self, session_id: int, summary: str, through: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE sessions SET summary = ?, summary_through = ? WHERE id = ?",
                (summary, through, session_id),
            )

    def previous_summaries(self, before_session: int, limit: int = 3
                           ) -> list[tuple[str, str]]:
        """(started_at, summary) for recent earlier sessions, oldest first."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT started_at, summary FROM sessions "
                "WHERE id < ? AND summary IS NOT NULL AND summary != '' "
                "ORDER BY id DESC LIMIT ?",
                (before_session, limit),
            ).fetchall()
        return [(r["started_at"], r["summary"]) for r in reversed(rows)]

    # --- turns ---

    def add_turn(self, session_id: int, role: str, content: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO turns (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content,
                 datetime.now().isoformat(timespec="seconds")),
            )
            return int(cur.lastrowid)

    def turns_after(self, session_id: int, after_id: int, limit: int = 20
                    ) -> list[dict]:
        """The most recent `limit` turns above `after_id`, in spoken order."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, role, content FROM turns "
                "WHERE session_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
                (session_id, after_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def turns_between(self, session_id: int, after_id: int, through_id: int
                      ) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, role, content FROM turns "
                "WHERE session_id = ? AND id > ? AND id <= ? ORDER BY id",
                (session_id, after_id, through_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def turn_count(self, session_id: int, after_id: int = 0) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM turns WHERE session_id = ? AND id > ?",
                (session_id, after_id),
            ).fetchone()
        return int(row["n"])

    # --- facts ---

    def remember(self, key: str, value: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute(
                "INSERT INTO facts (key, value, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key.strip().lower(), value.strip(), now, now),
            )

    def forget(self, key: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM facts WHERE key = ?", (key.strip().lower(),))
            return cur.rowcount > 0

    def facts(self) -> dict[str, str]:
        with self._conn() as c:
            rows = c.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}
