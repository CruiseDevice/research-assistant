from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    analysis: str # structured analysis
    sufficient: bool    # do we have enough to write the report?
    follow_up_query: str | None # if not sufficient, what to re-search


class SearchPlan(BaseModel):
    """Planner output: 1–3 focused, independent, web-searchable sub-queries
    into which a query is decomposed for parallel search.

    See PLAN_PHASE7_parallel.md (Phase 7P.1)."""

    subqueries: list[str] = Field(
        description="1–3 focused, independent, web-searchable sub-queries.",
    )
