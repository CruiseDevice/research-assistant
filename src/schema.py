from pydantic import BaseModel


class AnalysisResult(BaseModel):
    analysis: str # structured analysis
    sufficient: bool    # do we have enough to write the report?
    follow_up_query: str | None # if not sufficient, what to re-search
