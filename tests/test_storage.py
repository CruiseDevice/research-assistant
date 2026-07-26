"""Offline tests for ``src.storage.SessionStore``.

No network, no API keys, no graph. Each test gets a fresh DB in ``tmp_path``.
"""

from __future__ import annotations

import pytest

from src.storage import SessionStore


def _save_one(
    store: SessionStore,
    *,
    query: str = "who won the FIFA world cup in 2026?",
    iteration: int = 1,
    sufficient: bool = True,
    follow_up_query: str | None = None,
    search_results: str | None = "## Round 1\n**Spain beats Argentina** (https://example.com/1)",
    analysis: str | None = "Spain won 1-0.",
    report: str | None = "# Spain are 2026 champions\nSpain beat Argentina 1-0.",
) -> int:
    return store.save_session(
        query=query,
        iteration=iteration,
        sufficient=sufficient,
        follow_up_query=follow_up_query,
        search_results=search_results,
        analysis=analysis,
        report=report,
    )


def test_save_and_get_roundtrips_all_fields(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    sid = _save_one(store, follow_up_query=None)

    rec = store.get_session(sid)
    assert rec is not None
    assert rec.id == sid
    assert rec.query == "who won the FIFA world cup in 2026?"
    assert rec.iteration == 1
    assert rec.sufficient is True
    assert rec.follow_up_query is None
    assert rec.search_results is not None and "Spain beats Argentina" in rec.search_results
    assert rec.analysis == "Spain won 1-0."
    assert rec.report is not None and rec.report.startswith("# Spain are 2026 champions")
    # created_at is populated and ISO-8601-ish.
    assert rec.created_at and "T" in rec.created_at


def test_get_missing_returns_none(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    assert store.get_session(99999) is None


def test_sufficient_bool_roundtrips_through_integer_column(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    true_id = _save_one(store, sufficient=True)
    false_id = _save_one(store, sufficient=False)

    assert store.get_session(true_id).sufficient is True
    assert store.get_session(false_id).sufficient is False


def test_follow_up_query_none_survives_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    sid = _save_one(store, follow_up_query=None)
    rec = store.get_session(sid)
    assert rec is not None
    assert rec.follow_up_query is None


def test_follow_up_query_string_survives_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    sid = _save_one(store, follow_up_query="what was the final score?")
    rec = store.get_session(sid)
    assert rec is not None
    assert rec.follow_up_query == "what was the final score?"


def test_list_returns_newest_first(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    first = _save_one(store, query="first query")
    second = _save_one(store, query="second query")
    third = _save_one(store, query="third query")

    rows = store.list_sessions()
    assert [r.id for r in rows] == [third, second, first]
    assert [r.query for r in rows] == ["third query", "second query", "first query"]


def test_list_respects_limit(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    for i in range(5):
        _save_one(store, query=f"q{i}")

    rows = store.list_sessions(limit=2)
    assert len(rows) == 2
    # Newest two first.
    assert [r.query for r in rows] == ["q4", "q3"]


def test_list_empty_db_returns_empty_list(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    assert store.list_sessions() == []


def test_search_matches_query_case_insensitively(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    _save_one(store, query="Quantum Computing Basics")
    _save_one(store, query="something unrelated")

    rows = store.search_sessions("QUANTUM")
    assert len(rows) == 1
    assert rows[0].query == "Quantum Computing Basics"


def test_search_matches_analysis(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    _save_one(store, query="obscure query", analysis="the key insight was entanglement")

    rows = store.search_sessions("entanglement")
    assert len(rows) == 1


def test_search_matches_report(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    _save_one(store, query="obscure query", report="# Report\n\nQubits are fragile.")

    rows = store.search_sessions("qubits")
    assert len(rows) == 1


def test_search_no_matches_returns_empty(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    _save_one(store, query="alpha beta gamma")
    assert store.search_sessions("zzz-nope") == []


def test_search_respects_limit(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    for i in range(4):
        _save_one(store, query=f"common-term-{i}")
    rows = store.search_sessions("common-term", limit=2)
    assert len(rows) == 2


def test_delete_existing_returns_true_and_removes_row(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    sid = _save_one(store)

    assert store.delete_session(sid) is True
    assert store.get_session(sid) is None


def test_delete_missing_returns_false(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    assert store.delete_session(424242) is False


def test_schema_creation_is_idempotent(tmp_path):
    """Opening the same DB twice (as the CLI does on every call) must not error."""
    path = tmp_path / "s.db"
    store = SessionStore(path)
    _save_one(store)

    # A second store over the same file must be able to read the row back.
    store2 = SessionStore(path)
    rows = store2.list_sessions()
    assert len(rows) == 1


def test_db_parent_dir_is_created(tmp_path):
    """First run: the ~/.langgraph-research-assistant/-style parent dir may not exist yet."""
    nested = tmp_path / "deeply" / "nested" / "sessions.db"
    store = SessionStore(nested)
    sid = _save_one(store)
    assert nested.exists()
    assert store.get_session(sid) is not None


def test_default_db_path_is_under_home_dotdir():
    """When no path is passed, the store should target the user-level dotdir."""
    from pathlib import Path

    from src.storage import DEFAULT_DB_PATH

    assert DEFAULT_DB_PATH == Path.home() / ".langgraph-research-assistant" / "sessions.db"
