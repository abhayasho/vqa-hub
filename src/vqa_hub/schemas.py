
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List

class StepTrace(BaseModel):
    unit: str
    output: Dict[str, Any]

class RouterInfo(BaseModel):
    decision: str
    reason: str

class RunTrace(BaseModel):
    image: str
    question: str
    router: RouterInfo
    steps: List[StepTrace] = Field(default_factory=list)

class HubResult(BaseModel):
    answer: str
    trace: RunTrace
