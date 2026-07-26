"""SQLite-backed log of completed research sessions.

This is an *application-level* session log, not a LangGraph checkpointer.
After each run, the CLI persists the final graph state here so past research
sessions can be listed, searched, recalled, and deleted later. The graph
itself never touches this module.

The store uses the Python standard library ``sqlite3`` only — no extra
dependencies, no async, no ORM. The DB path defaults to
``~/.langgraph-research-assistant/sessions.db`` and is overridable per
instance (tests pass a ``tmp_path``).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_DB_PATH = Path.home() / ".langgraph-research-assistant" / "sessions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,           -- ISO-8601 UTC
    query           TEXT    NOT NULL,
    iteration       INTEGER NOT NULL,           -- number of search/analyze rounds
    sufficient      INTEGER NOT NULL,           -- 0/1 (analyst verdict)
    follow_up_query TEXT,
    search_results  TEXT,
    analysis        TEXT,
    report          TEXT
);
"""


@dataclass
class SessionSummary:
    """A short row used by ``--history`` and ``--search`` listings."""

    id: int
    created_at: str
    query: str
    iteration: int
    sufficient: bool


@dataclass
class SessionRecord(SessionSummary):
    """A full session row used by ``--show``. Adds the long text fields."""

    follow_up_query: Optional[str]
    search_results: Optional[str]
    analysis: Optional[str]
    report: Optional[str]


def _row_to_summary(row: sqlite3.Row) -> SessionSummary:
    return SessionSummary(
        id=row["id"],
        created_at=row["created_at"],
        query=row["query"],
        iteration=row["iteration"],
        sufficient=bool(row["sufficient"]),
    )


def _row_to_record(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        id=row["id"],
        created_at=row["created_at"],
        query=row["query"],
        iteration=row["iteration"],
        sufficient=bool(row["sufficient"]),
        follow_up_query=row["follow_up_query"],
        search_results=row["search_results"],
        analysis=row["analysis"],
        report=row["report"],
    )


class SessionStore:
    """Persists completed research sessions to a local SQLite database.

    Each public method opens its own short-lived connection (create + close),
    so the store is safe to call repeatedly from the one-shot CLI. The schema
    is created idempotently on every connection via ``CREATE TABLE IF NOT
    EXISTS``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        # ``None`` (the default for the CLI) falls back to DEFAULT_DB_PATH.
        self.db_path: Path = Path(db_path) if db_path else DEFAULT_DB_PATH

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Parent dir may not exist yet (e.g. first run, or a tmp_path).
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(_SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_session(
        self,
        *,
        query: str,
        iteration: int,
        sufficient: bool,
        follow_up_query: Optional[str],
        search_results: Optional[str],
        analysis: Optional[str],
        report: Optional[str],
    ) -> int:
        """Insert one session row and return the new row id."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO sessions
                   (created_at, query, iteration, sufficient, follow_up_query,
                    search_results, analysis, report)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    query,
                    iteration,
                    int(bool(sufficient)),
                    follow_up_query,
                    search_results,
                    analysis,
                    report,
                ),
            )
            return int(cur.lastrowid)

    def list_sessions(self, limit: int = 20) -> list[SessionSummary]:
        """Return the most recent sessions, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, created_at, query, iteration, sufficient
                   FROM sessions
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [_row_to_summary(r) for r in rows]

    def get_session(self, session_id: int) -> Optional[SessionRecord]:
        """Return the full record for one session, or ``None`` if absent."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id, created_at, query, iteration, sufficient,
                          follow_up_query, search_results, analysis, report
                   FROM sessions
                   WHERE id = ?""",
                (session_id,),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def search_sessions(self, term: str, limit: int = 20) -> list[SessionSummary]:
        """Case-insensitive substring search over query, analysis, and report."""
        like = f"%{term}%"
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, created_at, query, iteration, sufficient
                   FROM sessions
                   WHERE query LIKE ? ESCAPE '\\'
                      OR analysis LIKE ? ESCAPE '\\'
                      OR report LIKE ? ESCAPE '\\'
                   ORDER BY id DESC
                   LIMIT ?""",
                (like, like, like, limit),
            ).fetchall()
        return [_row_to_summary(r) for r in rows]

    def delete_session(self, session_id: int) -> bool:
        """Delete one session. Returns True if a row was actually removed."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,),
            )
            return cur.rowcount > 0
