
from __future__ import annotations
from typing import Optional
from transformers import pipeline, Pipeline

def make_captioner(model: str, device: Optional[object] = None) -> Pipeline:
    return pipeline("image-to-text", model=model, device=device)

def make_textgen(model: str, device: Optional[object] = None) -> Pipeline:
    return pipeline("text-generation", model=model, device=device)
