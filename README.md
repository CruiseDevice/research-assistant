# LangGraph Research Assistant

A command-line **multi-agent research assistant** built with [LangGraph](https://github.com/langchain-ai/langgraph). Give it a research query or a factual claim, and three agents work in sequence to search the web, analyze the results, and write a concise Markdown report.

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│ Researcher │ ──▶ │  Analyst   │ ──▶ │   Writer   │ ──▶ 📄 Report
│  (search)  │     │ (verify /  │     │ (markdown) │
│            │     │  insights) │     │            │
└────────────┘     └────────────┘     └────────────┘
```

- **Researcher** — searches the web and returns structured findings with sources.
- **Analyst** — verifies the claim (TRUE / FALSE / MIXED) or extracts key insights.
- **Writer** — turns the analysis into a concise Markdown report.

LangGraph provides explicit state management between steps, a clear graph visualization of the workflow, and an easy path to extension (loops, parallel nodes, memory).

---

## Requirements

- Python ≥ 3.10
- An LLM provider API key (**OpenAI** by default, or Anthropic)
- A web search API key ([Tavily](https://tavily.com) by default)

---

## Installation

### 1. Clone & enter the project

```bash
git clone https://github.com/CruiseDevice/research-assistant
cd research-assistant
```

### 2. Create & activate a virtual environment

Any Python ≥ 3.10 environment works. With **conda**:

```bash
conda create -n research python=3.12
conda activate research
```

Or with **venv**:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install the package (editable, with dev extras)

```bash
pip install -e ".[dev]"
# or, much faster with uv:
uv pip install -e ".[dev]"
```

This installs all dependencies **and** registers a `research` console script. Verify:

```bash
research --help        # once the CLI is built (Phase 5)
pip show research-assistant
```

---

## Configuration

Settings are loaded from a `.env` file (gitignored) and/or the process environment. Copy the template and fill in your keys:

```bash
cp .env.example .env
```

`.env` example:

```dotenv
# --- LLM ---
LLM_PROVIDER=openai                # "openai" (default) or "anthropic"
LLM_MODEL=gpt-4o

# Provider API keys
ANTHROPIC_API_KEY=
OPENAI_API_KEY=sk-...

# --- Search ---
SEARCH_PROVIDER=tavily             # "tavily" (default)
TAVILY_API_KEY=tvly-...
```

All variables are optional at import time — missing keys are validated at the point of use (the tool / agent that needs them). Variables can be overridden inline, e.g.:

```bash
LLM_MODEL=gpt-4o-mini research "What are quantum computers?"
```

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o` | Model name for the selected provider |
| `OPENAI_API_KEY` | *(empty)* | Required when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | *(empty)* | Required when `LLM_PROVIDER=anthropic` |
| `SEARCH_PROVIDER` | `tavily` | Search backend |
| `TAVILY_API_KEY` | *(empty)* | Required for web search |

Access settings from code:

```python
from src.config import settings
print(settings.llm_provider, settings.llm_model)
```

---

## How it works

The pipeline is a compiled LangGraph `StateGraph`. A shared `ResearchState` flows through three nodes; each node reads what it needs and returns a dict of state updates:

1. **`researcher`** — binds a [Tavily](https://tavily.com) search tool to the LLM, decides whether the input is a *query* or a *claim*, searches the web, and returns structured JSON:
   ```json
   { "findings": ["..."], "sources": ["https://..."], "input_type": "research_query" }
   ```
2. **`analyst`** — reads the findings. For a claim it returns a verdict (`TRUE` / `FALSE` / `MIXED`) and a source assessment; for a query it returns key insights:
   ```json
   { "input_type": "...", "insights": ["..."], "verdict": "TRUE", "source_assessment": "..." }
   ```
3. **`writer`** — turns the analysis into a Markdown report (≤ 500 words).

```python
from src.graph import build_research_graph

graph = build_research_graph()
result = graph.invoke({"user_input": "What are quantum computers?"})
print(result["report"])
```

---

## Development

```bash
# Install with dev extras (already done above)
pip install -e ".[dev]"

# Run tests
pytest
```

The stack: **LangGraph** (workflow) · **LangChain** (LLM abstraction) · **langchain-openai** / **langchain-anthropic** (providers) · **langchain-tavily** (search) · **pydantic-settings** (config).

---

## Roadmap

Planned extensions (see `PLAN.md` Phase 7):

- [ ] Feedback loop — Analyst can request the Researcher to search again
- [ ] Parallel research — multiple Researcher nodes with different queries
- [ ] Session memory — store reports in SQLite or a vector store
- [ ] Guardrails — validate agent outputs with Pydantic schemas + retry
- [ ] Web UI — wrap `run_research()` in FastAPI or Streamlit
- [ ] DuckDuckGo fallback — zero-API-key search option

---

## Notes

- **Secrets:** `.env` is gitignored. Never commit real API keys — store them in `.env` or the environment only.
- **Search backend:** we use the standalone [`langchain-tavily`](https://pypi.org/project/langchain-tavily/) package. Do **not** install `langchain-community` or `tavily-python` separately.
- See [`AGENTS.md`](./AGENTS.md) for the step-by-step build guide.
