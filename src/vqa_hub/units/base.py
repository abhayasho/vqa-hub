
from __future__ import annotations
from typing import Protocol, Dict, Any

class UnitModel(Protocol):
    name: str
    def run(self, **kwargs) -> Dict[str, Any]:
        ...
