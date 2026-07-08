"""Application configuration.

Loads secrets and settings from a local ``.env`` file (via python-dotenv) and
from the process environment, then exposes a single shared :data:`settings`
instance for the rest of the package to import.

Environment variables are mapped to fields case-insensitively, e.g.
``LLM_PROVIDER`` -> ``settings.llm_provider``.

Usage:
    from src.config import settings
    print(settings.llm_model)

Note: API keys default to an empty string so that importing this module always
succeeds (Phase 1 definition of done). The keys are validated later at their
point of use (see ``src/tools.py`` and ``src/agents.py``).
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load variables from .env into os.environ before instantiating Settings.
# Runs at import time so any module importing `settings` picks them up.
load_dotenv()


class Settings(BaseSettings):
    """Holds LLM and search configuration for the research assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o")

    # Provider API keys (empty by default; validated at use site)
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    # --- Search ---
    search_provider: str = Field(default="tavily")
    tavily_api_key: str = Field(default="")


# Single shared instance for the rest of the app to import.
settings = Settings()
