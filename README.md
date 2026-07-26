# LangGraph Research Assistant

A command-line **multi-agent research assistant** built with [LangGraph](https://github.com/langchain-ai/langgraph). Give it a research query or a factual claim, and three agents — search → analyze → report — collaboratively research it, with the analyst able to loop the searcher back for follow-up queries before writing a concise Markdown report.

```mermaid
flowchart TD
    Start([Start]) --> Search["search_agent<br/>(Tavily web search)"]
    Search --> Analyze["analyze_agent<br/>(analysis + sufficiency check)"]
    Analyze --> Route{"sufficient?<br/>or MAX_ITERATIONS reached?"}
    Route -->|"not sufficient"| Search
    Route -->|"sufficient / capped"| Report["report_agent<br/>(Markdown report)"]
    Report --> End([End])
```

- **search agent** — searches the web (Tavily) and summarizes the findings.
- **analyze agent** — produces a structured analysis and judges whether the evidence is sufficient, or requests a follow-up search.
- **report agent** — turns the analysis into a concise Markdown report.

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
| `SEARCH_MODE` | `parallel` | `parallel` (planner + fan-out, default) or `react` (legacy ReAct summarizer) |
| `TAVILY_API_KEY` | *(empty)* | Required for web search |
| `SESSION_DB_PATH` | *(empty)* | SQLite session-log path. Empty uses `~/.langgraph-research-assistant/sessions.db` |

Access settings from code:

```python
from src.config import settings
print(settings.llm_provider, settings.llm_model)
```

---

## Usage

Run a research query (one-shot or interactive):

```bash
research                              # interactive prompt
research "who won the FIFA world cup in 2026?"   # one-shot
research "..." --no-save              # run without logging the session
```

Each completed run is logged to a local SQLite database so you can recall past
research sessions. Reading and managing the log needs no API keys:

```bash
research --history                    # list recent sessions (newest first)
research --history --limit 50
research --show 17                    # print a past session's report
research --search "quantum"           # substring search over query/analysis/report
research --delete 17                  # delete a session
```

---

## How it works

The pipeline is a compiled LangGraph `StateGraph` exposed as the module-level `graph`. A shared `ResearchState` (keyed on `query`) flows through three nodes; the analyst can loop the searcher back for a follow-up, capped at `MAX_ITERATIONS` (=2) rounds:

1. **`search_agent`** — a ReAct agent (`create_react_agent`) with a [Tavily](https://tavily.com) search tool bound; gathers web results and summarizes the key facts, titles, and source URLs. Results are accumulated across rounds, each tagged `## Round N`.
2. **`analyze_agent`** — reads the accumulated results (treated as data inside `<search_results>` tags to limit prompt injection) and returns structured output:
   ```json
   { "analysis": "...", "sufficient": true, "follow_up_query": null }
   ```
3. **Routing** — if `sufficient` is `false` and the round count is under `MAX_ITERATIONS`, it loops back to `search_agent` with `follow_up_query`; otherwise it proceeds.
4. **`report_agent`** — turns the analysis into a concise Markdown report.

```python
from src.graph import graph

result = graph.invoke({"query": "What are quantum computers?"})
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
