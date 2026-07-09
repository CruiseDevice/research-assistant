"""Shared graph state definition."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    """TypedDict-style state passed between agents."""

    query: str
    messages: Annotated[list, add_messages]
    search_results: str
    analysis: str
    report: str
