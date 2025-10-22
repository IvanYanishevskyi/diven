from typing import Literal, Optional, List
from pydantic import BaseModel, Field

class Plan(BaseModel):
    intent: str = Field(description="What the user wants, one line")
    reasoning: str = Field(description="Key steps in plain language")
    chart: Literal["bar", "line", "hist", "pie", "table","heatmap"] = "bar"
    pandas_code: str = Field(description="Minimal pandas code using df; MUST assign to variable 'result'")
    answer_hint: Optional[str] = Field(default=None, description="Optional natural-language answer/summary")

class AnalysisResponse(BaseModel):
    question: str
    intent: str
    reasoning: str
    chart: Literal["bar", "line", "hist", "pie", "table","heatmap"]
    chart_png_base64: str
    preview_rows: int
    preview_cols: int
    used_columns: Optional[List[str]] = None
    pandas_code: str
    answer_hint: Optional[str] = None