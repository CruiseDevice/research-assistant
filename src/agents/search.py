"""Search agent: gathers web results via Tavily and summarizes them."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.llm import llm
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


def search_node(state: ResearchState):
    result = search_agent.invoke({
        "messages": [HumanMessage(content=state["query"])]
    })
    final_message = result["messages"][-1]  # last AIMessage = the summary
    return {
        "messages": result["messages"],
        "search_results": final_message.content,
    }
