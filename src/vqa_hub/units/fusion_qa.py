from __future__ import annotations
from typing import Dict, Any
from .text.qa_small import ExtractiveQA
from .base import UnitModel

class FusionQA(UnitModel):
    def __init__(self, qa_model: ExtractiveQA):
        self.name = "fusion:extractive_qa"
        self.qa_model = qa_model

    def build_context(self, caption: str, ocr_text: str | None = None) -> str:
        parts = [f"Caption: {caption}"]
        if ocr_text:
            parts.append(f"On-image text: {ocr_text}")
        return "  ".join(parts)

    def run(self, caption: str, question: str, ocr_text: str | None = None) -> Dict[str, Any]:
        context = self.build_context(caption, ocr_text)
        out = self.qa_model.run(context=context, question=question)
        return {"answer_text": out["answer"], "score": out["score"], "context": context}
