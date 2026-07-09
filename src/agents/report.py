"""Report agent: writes the final Markdown report from the analysis."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import llm
from src.state import ResearchState

REPORT_SYSTEM = SystemMessage(
    content="""
You are a technical writer. Given the query and the analyst's notes,
write a concise Markdown report. Use headers, bullets, and cite sources
where possible.
Keep it under ~300 words unless the topic clearly needs more depth.
"""
)


def report_agent(state: ResearchState):
    state_query = state.get("query", "")
    state_analysis = state.get("analysis", "")
    if not state_analysis:
        return {
            "report": "No analysis available to report on.",
            "messages": [],
        }
    prompt = (
        f"Query: {state_query}\n\n"
        f"Analysis:\n{state_analysis}\n\n"
        "Write the final Markdown report."
    )
    messages = [REPORT_SYSTEM, HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return {
        "report": response.content,
        "messages": [response],
    }
