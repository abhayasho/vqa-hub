
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RouterDecision:
    path: str
    reason: str

class Router:
    def __init__(self, use_big_fallback: bool, complex_cues: list[str], prefer_direct_vqa: bool, ocr_text_min_chars: int):
        self.use_big_fallback = use_big_fallback
        self.complex_cues = [c.lower() for c in complex_cues]
        self.prefer_direct_vqa = prefer_direct_vqa
        self.ocr_text_min_chars = int(ocr_text_min_chars)

    def decide(self, question: str) -> RouterDecision:
        q = question.lower()
        if "text" in q or "say" in q or "read" in q:
            return RouterDecision(path="caption -> ocr -> fusion -> maybe_big", reason="text-reading query")
        if any(c in q for c in self.complex_cues):
            return RouterDecision(path="caption -> vqa -> maybe_big", reason="complex query")
        if self.prefer_direct_vqa:
            return RouterDecision(path="caption -> vqa", reason="prefer direct vqa")
        return RouterDecision(path="caption -> fusion", reason="simple query")
