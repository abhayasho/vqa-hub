
from __future__ import annotations
from typing import Dict, Any
from transformers import Pipeline
from ..base import UnitModel

class TextGenerator(UnitModel):
    def __init__(self, pipe: Pipeline, model_name: str, max_new_tokens: int = 64):
        self.name = f"textgen:{model_name}"
        self.pipe = pipe
        self.max_new_tokens = max_new_tokens

    def run(self, prompt: str) -> Dict[str, Any]:
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            temperature=0.7,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            pad_token_id=50256,
            eos_token_id=50256,
        )
        text = outputs[0]["generated_text"][len(prompt):].strip()
        return {"text": text, "raw": outputs}
