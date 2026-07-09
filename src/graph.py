"""Builds and compiles the research graph."""

from langgraph.graph import END, START, StateGraph

from src.agents import analyze_agent, report_agent, search_node
from src.state import ResearchState

builder = StateGraph(ResearchState)

builder.add_node("search_agent", search_node)
builder.add_node("analyze_agent", analyze_agent)
builder.add_node("report_agent", report_agent)

builder.add_edge(START, "search_agent")
builder.add_edge("search_agent", "analyze_agent")
builder.add_edge("analyze_agent", "report_agent")
builder.add_edge("report_agent", END)

graph = builder.compile()
