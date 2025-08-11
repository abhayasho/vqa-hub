
from __future__ import annotations
try:
    from langchain_community.llms import Ollama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

def make_ollama(model: str):
    if not HAS_OLLAMA:
        raise RuntimeError("langchain_community/Ollama not installed or ollama daemon not running.")
    return Ollama(model=model)
