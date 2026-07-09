"""Shared LLM instance used by all agents."""

from langchain_openai import ChatOpenAI

from src.config import settings

# Single shared instance so every agent uses the same model/temperature.
llm = ChatOpenAI(model=settings.llm_model, temperature=0.2)
