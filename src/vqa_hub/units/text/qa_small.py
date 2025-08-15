from __future__ import annotations
from typing import Dict, Any
from transformers import pipeline
from ..base import UnitModel

class ExtractiveQA(UnitModel):
    def __init__(self, model: str = "deepset/roberta-base-squad2", device=None):
        self.name = f"qa:{model}"
        self.pipe = pipeline("question-answering", model=model, device=device)

    def run(self, context: str, question: str) -> Dict[str, Any]:
        out = self.pipe(question=question, context=context)
        return {"answer": out.get("answer","").strip(), "score": float(out.get("score",0.0)), "raw": out}
