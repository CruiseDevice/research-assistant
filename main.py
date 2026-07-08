from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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

# Bind tools to a model instance that can emit tool calls
search_llm = llm.bind_tools(tools)


SEARCH_SYSTEM = SystemMessage(
    content="""
    You are a research search agent.
    Given the user's query, use the TavilySearch tool to find the most relevant, current web
    results.
    Return a concise summary of the search results including key facts, source URLs, and titles.
    Do not analyze or conclude-only gather and summarize what was found.
    """
)


def search_agent(state: ResearchState):
    messages = [SEARCH_SYSTEM, HumanMessage(content=state["query"])]
    response = search_llm.invoke(messages)

    # If the LLM emitted a tool call, execute it 
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = tavily_search.invoke(tool_call["args"])
        # feed the tool result back, so llm can summarize it

        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
        final_response = search_llm.invoke(
            messages + [response, tool_message]
        )

        return {
            "messages": [response, tool_message, final_response],
            "search_results": final_response.content,
        }
    
    # Fallback if no tool was called
    return {
        "search_results": response.content, "messages": [response]
    }

def analyze_agent(state: ResearchState):
    pass


def report_agent(state: ResearchState):
    pass


builder = StateGraph(ResearchState)

builder.add_node("search_agent", search_agent)
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