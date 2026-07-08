from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from src.config import settings


class ResearchState(TypedDict, total=False):
    """
    TypedDict-style state passed between agents.
    """
    query: str
    messages: Annotated[list, add_messages]
    search_results: str
    analysis: str
    report: str


llm = ChatOpenAI(model=settings.llm_model, temperature=0.2)

# Tavily search tool: returns ranked results with title, url, content, score
tavily_search = TavilySearch(max_results=5, search_depth="advanced")
tools = [tavily_search]

SEARCH_SYSTEM = SystemMessage(
    content="""
    You are a research search agent.
    Given the user's query, use the TavilySearch tool to find the most relevant, current web
    results.
    Return a concise summary of the search results including key facts, source URLs, and titles.
    Do not analyze or conclude-only gather and summarize what was found.
    """
)

search_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SEARCH_SYSTEM.content
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


def analyze_agent(state: ResearchState):
    pass


def report_agent(state: ResearchState):
    pass


builder = StateGraph(ResearchState)

builder.add_node("search_agent", search_node)
builder.add_node("analyze_agent", analyze_agent)
builder.add_node("report_agent", report_agent)


builder.add_edge(START, "search_agent")
builder.add_edge("search_agent", "analyze_agent")
builder.add_edge("analyze_agent", "report_agent")
builder.add_edge("report_agent", END)

graph = builder.compile()


if __name__ == "__main__":
    query = input("Enter a research query or factual claim:")
    result = graph.invoke({
        "query": query
    })
    print(result)