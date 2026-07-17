"""Shared LLM instance used by all agents.

Selects the LangChain chat-model class from ``settings.llm_provider``
(``openai`` or ``anthropic``) and wires up the matching API key.

Previously this module hardcoded ``ChatOpenAI`` and ignored the provider
config, which silently made ``LLM_PROVIDER=anthropic`` dead code (see the
note in ``PRODUCTION_ANSWER.md``). The selected provider's API key is
validated here, at the point of use, so importing :mod:`src.config` never
fails while a misconfigured run fails fast with a clear message.
"""

from __future__ import annotations

from src.config import settings


def _build_llm():
    """Construct the shared chat model for the configured provider."""
    provider = settings.llm_provider.strip().lower()

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        # Imported lazily so an OpenAI-only run doesn't need the package.
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=0.2,
            api_key=settings.anthropic_api_key,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai (default) but OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={settings.llm_provider!r}. "
        "Use 'openai' or 'anthropic'."
    )


# Single shared instance so every agent uses the same model/temperature.
llm = _build_llm()
