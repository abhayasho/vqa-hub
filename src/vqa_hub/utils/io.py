
from __future__ import annotations
from pathlib import Path
from PIL import Image

def load_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(p).convert("RGB")
