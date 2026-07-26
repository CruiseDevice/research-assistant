"""Command-line entrypoint.

Run via the ``research`` console script (see ``pyproject.toml``),
via ``python -m src.main``, or via the root ``main.py`` shim.

Usage:
    research                              # interactive prompt
    research "who won the 2026 world cup" # one-shot query
    research --history [--limit N]        # list recent sessions
    research --show 17                    # print a past session's report
    research --search "quantum"           # substring search over past sessions
    research --delete 17                  # delete a session
    research "..." --no-save              # run without logging the session
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from src.config import settings
from src.storage import SessionStore, SessionSummary


def _build_store() -> SessionStore:
    """Construct the session store from the configured DB path (if any)."""
    return SessionStore(settings.session_db_path or None)


def _format_summary(row: SessionSummary) -> str:
    flag = "ok" if row.sufficient else "..."
    return (f"#{row.id:<4} {row.created_at}  [{flag}]  "
            f"({row.iteration} round{'s' if row.iteration != 1 else ''})  {row.query}")


def run_read_command(args: argparse.Namespace, store: SessionStore, out=sys.stdout) -> int:
    """Dispatch the read/delete flags. Returns a process exit code.

    Kept pure (no ``sys.exit``) and takes the store as an argument so it can
    be unit-tested against a stub store without touching the graph or network.
    """
    if args.history:
        rows = store.list_sessions(limit=args.limit)
        if not rows:
            print("No saved sessions yet.", file=out)
            return 0
        for row in rows:
            print(_format_summary(row), file=out)
        return 0

    if args.show is not None:
        rec = store.get_session(args.show)
        if rec is None:
            print(f"No session with id {args.show}.", file=out)
            return 1
        header = (f"# Session #{rec.id}  ({rec.created_at})\n"
                  f"query: {rec.query}\n"
                  f"rounds: {rec.iteration}   sufficient: {rec.sufficient}")
        if rec.follow_up_query:
            header += f"\nfollow-up: {rec.follow_up_query}"
        print(header, file=out)
        print("\n--- report ---", file=out)
        print(rec.report or "(no report)", file=out)
        return 0

    if args.search:
        rows = store.search_sessions(args.search, limit=args.limit)
        if not rows:
            print(f"No sessions matching {args.search!r}.", file=out)
            return 0
        for row in rows:
            print(_format_summary(row), file=out)
        return 0

    if args.delete is not None:
        ok = store.delete_session(args.delete)
        if ok:
            print(f"Deleted session #{args.delete}.", file=out)
            return 0
        print(f"No session with id {args.delete}.", file=out)
        return 1

    # No read flag set — caller (main) should run research instead.
    return 0


def _run_research(query: Optional[str], store: SessionStore, save: bool) -> None:
    """Run the graph for one query and optionally persist the resulting session."""
    # Imported lazily so the read commands (--history / --show / --search /
    # --delete) work without the langgraph stack or API keys being available,
    # and so the CLI dispatch is unit-testable in isolation.
    from src.graph import graph

    if query is None:
        query = input("Enter a research query or factual claim:")
    result = graph.invoke({"query": query})
    print(result.get("report", result))

    if not save:
        return
    # Persisting is best-effort: never let it break the research output.
    try:
        sid = store.save_session(
            query=query,
            iteration=int(result.get("iteration", 0) or 0),
            sufficient=bool(result.get("sufficient", False)),
            follow_up_query=result.get("follow_up_query"),
            search_results=result.get("search_results"),
            analysis=result.get("analysis"),
            report=result.get("report"),
        )
        print(f"[session #{sid} saved]", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — best-effort log, surface to user
        print(f"[warning: could not save session: {exc}]", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="research",
        description="LangGraph research assistant.",
    )
    p.add_argument("query", nargs="?", help="research query (omit for interactive prompt)")
    p.add_argument("--history", action="store_true", help="list recent sessions")
    p.add_argument("--show", type=int, metavar="ID", help="print a past session's report")
    p.add_argument("--search", metavar="TERM", help="substring search over past sessions")
    p.add_argument("--delete", type=int, metavar="ID", help="delete a session")
    p.add_argument("--limit", type=int, default=20, help="max rows for --history/--search (default 20)")
    p.add_argument("--no-save", action="store_true", help="don't log this run to the session DB")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    store = _build_store()

    read_flags_active = any([
        args.history,
        args.show is not None,
        bool(args.search),
        args.delete is not None,
    ])
    if read_flags_active:
        sys.exit(run_read_command(args, store))

    _run_research(args.query, store, save=not args.no_save)


if __name__ == "__main__":
    main()
