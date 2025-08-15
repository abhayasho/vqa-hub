from __future__ import annotations
from typing import Dict, Any
from PIL import Image
from transformers import Pipeline
from ..base import UnitModel

class ImageCaptioner(UnitModel):
    def __init__(self, pipe: Pipeline, model_name: str):
        self.name = f"captioner:{model_name}"
        self.pipe = pipe

    def run(self, image: Image.Image, max_new_tokens: int = 30) -> Dict[str, Any]:
        # For transformers>=4.40, pass decoding controls via generate_kwargs
        outputs = self.pipe(
            image,
            max_new_tokens=max_new_tokens,
            generate_kwargs=dict(
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                min_length=8,
            ),
        )
        caption = outputs[0]["generated_text"].strip()
        return {"caption": caption, "raw": outputs}
