
from __future__ import annotations
from typing import Dict, Any
from PIL import Image
from transformers import Pipeline
from ..base import UnitModel

class BLIPVQA(UnitModel):
    def __init__(self, pipe: Pipeline, model_name: str):
        self.name = f"vqa:{model_name}"
        self.pipe = pipe

    def run(self, image: Image.Image, question: str, max_new_tokens: int = 20) -> Dict[str, Any]:
        out = self.pipe(image=image, question=question, max_new_tokens=max_new_tokens)
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            ans = out[0]["generated_text"].strip()
        elif isinstance(out, list) and out and "answer" in out[0]:
            ans = out[0]["answer"].strip()
        else:
            ans = str(out).strip()
        return {"answer": ans, "raw": out}
