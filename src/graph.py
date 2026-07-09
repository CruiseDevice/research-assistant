"""Builds and compiles the research graph."""

from langgraph.graph import END, START, StateGraph

from src.agents import analyze_agent, report_agent, search_node
from src.state import ResearchState

MAX_ITERATIONS = 2

def route_after_analysis(state: ResearchState):
    sufficient = state.get("sufficient", False)
    iteration = state.get("iteration", 0)

    if sufficient:
        print(f"[route] sufficient=True -> report_agent (after {iteration} round(s))")
        return "report_agent"
    elif iteration >= MAX_ITERATIONS:
        print(f"[route] iteration cap ({MAX_ITERATIONS}) reached -> report_agent")
        return "report_agent" # force-finish, don't loop forever

    print(f"[route] iteration={iteration}, not sufficient -> search_agent (loop back)")
    return "search_agent"

builder = StateGraph(ResearchState)

builder.add_node("search_agent", search_node)
builder.add_node("analyze_agent", analyze_agent)
builder.add_node("report_agent", report_agent)

builder.add_edge(START, "search_agent")
builder.add_edge("search_agent", "analyze_agent")
builder.add_conditional_edges(
    "analyze_agent",
    route_after_analysis,
    ["search_agent", "report_agent"]
)
builder.add_edge("report_agent", END)

graph = builder.compile()
