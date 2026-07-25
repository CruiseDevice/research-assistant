"""Search agent: gathers web results via Tavily and summarizes them."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from src.config import settings
from src.llm import llm
from src.schema import SearchPlan
from src.state import ResearchState
from src.tools import tools

SEARCH_SYSTEM = SystemMessage(
    content="""
You are a research search agent.
Given the user's query, use the TavilySearch tool to find the most relevant, current web
results.
Return a concise summary of the search results including key facts, source URLs, and titles.
Do not analyze or conclude - only gather and summarize what was found.
"""
)

# create_react_agent binds the tools itself and runs the full
# model -> (all) tool calls -> model loop until no more tool calls.
search_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SEARCH_SYSTEM.content,
)


# --- Planner (Phase 7P.1) ---------------------------------------------------
# Decomposes a query into 1–3 independent, web-searchable sub-queries for the
# parallel fan-out. Wired into search_node (Phase 7P.3); also unit-testable.

PLANNER_SYSTEM = SystemMessage(
    content="""
You are a research planning agent.
Given a query, decompose it into 1–3 focused, independent, web-searchable
sub-queries that together cover the query's key facets.
Rules:
- Never return more than 3 sub-queries.
- If the query is simple or single-faceted, return a single-element list with
  the query itself (lightly reworded only if that improves searchability).
- Each sub-query must be self-contained and searchable as-is on the web.
- Do not analyze or answer the query; only produce the plan.
"""
)

# Structured output: a parsed SearchPlan (subqueries: list[str]).
planner_llm = llm.with_structured_output(SearchPlan)


def plan_search(query: str) -> SearchPlan:
    """Decompose ``query`` into 1–3 web-searchable sub-queries.

    Non-fatal by design (locked decision #6): if the planner raises or returns
    an unusable result, fall back to a single-element plan of the original
    query so the graph can never crash here. Sub-query dedupe is deferred to
    the fan-out (Phase 7P.2).
    """
    try:
        plan = planner_llm.invoke([PLANNER_SYSTEM, HumanMessage(content=query)])
    except Exception as exc:  # broad on purpose: planner must never crash the run
        print(f"[plan] planner failed ({exc!r}); falling back to [query]")
        return SearchPlan(subqueries=[query])

    # Guarantee the 1–3 usable-strings contract even if the model misbehaves.
    subs = [s for s in (plan.subqueries or []) if s and s.strip()]
    if not subs:
        print("[plan] planner returned no usable sub-queries; falling back to [query]")
        return SearchPlan(subqueries=[query])
    return SearchPlan(subqueries=subs[:3])


# --- Fan-out (Phase 7P.2) ---------------------------------------------------
# Runs Tavily concurrently across the planned sub-queries. Each call is
# isolated: a single failing sub-query is dropped, never propagated. Wired into
# search_node (Phase 7P.3); also unit-testable.

# Dedicated instance: max_results=3 so <=3 sub-queries stay within a bounded
# context. The global tavily_search (max_results=5) used by the ReAct
# search_agent is left untouched.
parallel_search_tool = TavilySearch(max_results=3, search_depth="advanced")


def fan_out_search(subqueries: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Search every sub-query concurrently via Tavily.

    Returns ``(subquery, tavily_response)`` pairs for the sub-queries that
    succeeded, in completion order. Identical sub-queries are deduped (naive
    exact match, order-preserving). A failing call (timeout, network,
    rate-limit) is logged and skipped -- the round never crashes.

    The empty-input case returns ``[]``; the caller (search_node, 7P.3) owns
    the ``[]`` -> ``[query]`` fallback, since only it has the original query.
    """
    # Dedupe identical sub-queries, preserving first-seen order.
    seen: set[str] = set()
    unique: list[str] = []
    for sq in subqueries:
        if sq not in seen:
            seen.add(sq)
            unique.append(sq)
    if not unique:
        return []

    pairs: list[tuple[str, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(len(unique), 3)) as pool:
        future_to_sq = {
            pool.submit(parallel_search_tool.invoke, {"query": sq}): sq
            for sq in unique
        }
        for fut in as_completed(future_to_sq):
            sq = future_to_sq[fut]
            try:
                result = fut.result()  # re-raises the worker's exception
            except Exception as exc:  # per-future isolation; never crash the round
                print(f"[search] sub-query failed, skipping: {sq!r} ({exc!r})")
                continue
            if not result:
                continue
            pairs.append((sq, result))
    return pairs


def _accumulate_round(previous: str, iteration: int, body: str) -> str:
    """Append this round's ``body`` under a ``## Round N`` header.

    Keeps the exact round labeling the analyst's prompt relies on ("synthesize
    across rounds separated by '---' and labeled '## Round N'"). Shared by both
    the parallel and ReAct paths so labeling is identical.
    """
    round_block = f"## Round {iteration + 1}\n\n{body}"
    return f"{previous}\n\n---\n\n{round_block}" if previous else round_block


def _format_subquery(sq: str, response: dict[str, Any]) -> str:
    """Render one sub-query's Tavily response as labeled Markdown snippets."""
    results = response.get("results", []) if isinstance(response, dict) else []
    parts = [f"### Sub-query: {sq}"]
    for r in results:
        title = (r.get("title") or "").strip() or "(untitled)"
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        parts.append(f"**{title}** ({url})" if url else f"**{title}**")
        if content:
            parts.append(content)
    if len(parts) == 1:  # header only, no results
        parts.append("(no results)")
    return "\n\n".join(parts)


def search_node(state: ResearchState):
    """Gather web results for the round and accumulate into ``search_results``.

    Branches on ``settings.search_mode`` (decision #7):
    - ``"parallel"`` (default): plan the active query into sub-queries, fan out
      to Tavily concurrently, combine the raw per-sub-query snippets.
    - ``"react"``: legacy single-shot ReAct search + summary (A/B fallback).
    The analyst's ``follow_up_query`` (if any) takes precedence over the
    original ``query`` as the thing we plan/search.
    """
    query = state.get("query", "")
    follow_up_query = state.get("follow_up_query", "")
    iteration = state.get("iteration", 0)
    previous = state.get("search_results", "")

    active_query = follow_up_query if follow_up_query else query
    label = "follow-up" if follow_up_query else "query"

    mode = settings.search_mode.strip().lower()
    if mode not in ("parallel", "react"):
        raise ValueError(
            f"Unknown SEARCH_MODE={settings.search_mode!r}. "
            "Use 'parallel' or 'react'."
        )

    # --- ReAct (A/B fallback) ------------------------------------------------
    if mode == "react":
        print(f"[search] round {iteration + 1} | mode=react | {label}: {active_query}")
        result = search_agent.invoke({"messages": [HumanMessage(content=active_query)]})
        body = result["messages"][-1].content
        return {
            "messages": result["messages"],
            "search_results": _accumulate_round(previous, iteration, body),
        }

    # --- Parallel (default) --------------------------------------------------
    print(f"[search] round {iteration + 1} | mode=parallel | {label}")
    plan = plan_search(active_query)
    print(f"[search] round {iteration + 1} | planned: {plan.subqueries}")

    pairs = fan_out_search(plan.subqueries)
    if pairs:
        body = "\n\n".join(_format_subquery(sq, resp) for sq, resp in pairs)
    else:
        body = "(search returned no usable results this round)"

    return {
        "messages": [
            AIMessage(
                content=(
                    f"[round {iteration + 1}] parallel search: "
                    f"{len(plan.subqueries)} planned, {len(pairs)} succeeded"
                )
            )
        ],
        "search_results": _accumulate_round(previous, iteration, body),
    }
