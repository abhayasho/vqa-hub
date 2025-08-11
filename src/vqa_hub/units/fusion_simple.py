
from __future__ import annotations
from typing import Dict, Any
from .text.gpt2_textgen import TextGenerator
from .base import UnitModel

class SimpleVQAFusion(UnitModel):
    def __init__(self, text_model: TextGenerator):
        self.name = "fusion:simple_vqa_via_text"
        self.text_model = text_model

    def build_prompt(self, caption: str, question: str) -> str:
        system = (
            "You answer questions about an image. "
            "You are given a caption that describes the image. "
            "Be concise and accurate. If unknown, say you are not sure.\n\n"
        )
        return f"{system}Caption: {caption}\nQuestion: {question}\nAnswer: "

    def run(self, caption: str, question: str) -> Dict[str, Any]:
        prompt = self.build_prompt(caption, question)
        out = self.text_model.run(prompt=prompt)
        return {"answer_text": out["text"], "prompt": prompt}
