"""Offline tests for the CLI read-command dispatch in ``src.main``.

These tests exercise ``run_read_command`` against a stub store and capture
stdout via an ``io.StringIO`` — no real DB, no graph, no network, no API keys.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass

import pytest

from src.main import _build_parser, run_read_command


@dataclass
class _StubSummary:
    id: int
    created_at: str
    query: str
    iteration: int
    sufficient: bool


@dataclass
class _StubRecord:
    id: int
    created_at: str
    query: str
    iteration: int
    sufficient: bool
    follow_up_query: str | None
    search_results: str | None
    analysis: str | None
    report: str | None


class _StubStore:
    """Records what the CLI asked for and returns canned data."""

    def __init__(self) -> None:
        self.list_calls: list[int] = []
        self.get_calls: list[int] = []
        self.search_calls: list[tuple[str, int]] = []
        self.delete_calls: list[int] = []

    def list_sessions(self, limit: int = 20):
        self.list_calls.append(limit)
        return [
            _StubSummary(id=3, created_at="2026-07-26T10:00:00+00:00", query="third",
                         iteration=2, sufficient=True),
            _StubSummary(id=2, created_at="2026-07-25T10:00:00+00:00", query="second",
                         iteration=1, sufficient=False),
        ]

    def get_session(self, session_id: int):
        self.get_calls.append(session_id)
        if session_id == 17:
            return _StubRecord(
                id=17, created_at="2026-07-26T10:00:00+00:00",
                query="who won the 2026 world cup?", iteration=1, sufficient=True,
                follow_up_query=None, search_results="raw", analysis="Spain won.",
                report="# Spain are 2026 champions\nSpain beat Argentina 1-0.",
            )
        return None

    def search_sessions(self, term: str, limit: int = 20):
        self.search_calls.append((term, limit))
        return [_StubSummary(id=5, created_at="2026-07-26T10:00:00+00:00",
                             query="quantum query", iteration=1, sufficient=True)]

    def delete_session(self, session_id: int):
        self.delete_calls.append(session_id)
        return session_id == 17


def _parse(argv: list[str]) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _run(argv: list[str], store: _StubStore) -> tuple[int, str]:
    """Run a read command with the given argv; return (exit_code, captured_stdout)."""
    buf = io.StringIO()
    code = run_read_command(_parse(argv), store, out=buf)
    return code, buf.getvalue()


# --- --history ---------------------------------------------------------------

def test_history_calls_list_and_prints_rows():
    store = _StubStore()
    code, out = _run(["--history"], store)

    assert code == 0
    assert store.list_calls == [20]
    assert "third" in out
    assert "second" in out
    # Newest first.
    assert out.index("third") < out.index("second")


def test_history_passes_limit_through():
    store = _StubStore()
    _run(["--history", "--limit", "5"], store)
    assert store.list_calls == [5]


def test_history_empty_shows_message():
    store = _StubStore()
    store.list_sessions = lambda limit=20: []  # type: ignore[assignment]
    code, out = _run(["--history"], store)
    assert code == 0
    assert "No saved sessions" in out


# --- --show ------------------------------------------------------------------

def test_show_existing_prints_report():
    store = _StubStore()
    code, out = _run(["--show", "17"], store)

    assert code == 0
    assert store.get_calls == [17]
    assert "Session #17" in out
    assert "who won the 2026 world cup?" in out
    assert "--- report ---" in out
    assert "Spain are 2026 champions" in out


def test_show_missing_returns_nonzero_exit_and_message():
    store = _StubStore()
    code, out = _run(["--show", "999"], store)

    assert code == 1
    assert store.get_calls == [999]
    assert "No session with id 999" in out


# --- --search ----------------------------------------------------------------

def test_search_calls_search_and_prints_matches():
    store = _StubStore()
    code, out = _run(["--search", "quantum"], store)

    assert code == 0
    assert store.search_calls == [("quantum", 20)]
    assert "quantum query" in out


def test_search_no_matches_shows_message():
    store = _StubStore()
    store.search_sessions = lambda term, limit=20: []  # type: ignore[assignment]
    code, out = _run(["--search", "nothing"], store)
    assert code == 0
    assert "No sessions matching 'nothing'" in out


# --- --delete ----------------------------------------------------------------

def test_delete_existing_returns_zero_and_confirms():
    store = _StubStore()
    code, out = _run(["--delete", "17"], store)

    assert code == 0
    assert store.delete_calls == [17]
    assert "Deleted session #17" in out


def test_delete_missing_returns_nonzero():
    store = _StubStore()
    code, out = _run(["--delete", "999"], store)

    assert code == 1
    assert store.delete_calls == [999]
    assert "No session with id 999" in out


# --- dispatch hygiene --------------------------------------------------------

def test_no_read_flags_does_not_call_any_store_method():
    """If none of the read flags are set, run_read_command must be a no-op
    (it's main()'s job to then run research)."""
    store = _StubStore()
    code, out = _run([], store)

    assert code == 0
    assert out == ""
    assert store.list_calls == []
    assert store.get_calls == []
    assert store.search_calls == []
    assert store.delete_calls == []


def test_parser_accepts_positional_query():
    """The bare one-shot form (positional query) must parse without error."""
    ns = _parse(["some query text"])
    assert ns.query == "some query text"
    assert ns.history is False
    assert ns.show is None
    assert ns.search is None
    assert ns.delete is None


def test_parser_no_args_yields_none_query():
    """The bare interactive form: nothing on the command line."""
    ns = _parse([])
    assert ns.query is None
    assert ns.no_save is False
