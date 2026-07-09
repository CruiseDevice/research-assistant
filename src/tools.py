"""Tools available to the agents."""

from langchain_tavily import TavilySearch

# Tavily search tool: returns ranked results with title, url, content, score.
tavily_search = TavilySearch(max_results=5, search_depth="advanced")
tools = [tavily_search]
