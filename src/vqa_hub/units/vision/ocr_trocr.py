
from __future__ import annotations
from typing import Dict, Any
from PIL import Image
from transformers import pipeline
from ..base import UnitModel

class OCRTrOCR(UnitModel):
    def __init__(self, model: str = "microsoft/trocr-base-printed", device=None):
        self.name = f"ocr:{model}"
        self.pipe = pipeline("image-to-text", model=model, device=device)

    def run(self, image: Image.Image) -> Dict[str, Any]:
        out = self.pipe(image)
        text = out[0].get("generated_text", "").strip() if out else ""
        return {"text": text, "raw": out}
