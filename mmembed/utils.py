from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

@dataclass
class GenerationInput:
    instruction: str = ""
    text: str = ""
    img_paths: tp.List[str] = field(default_factory=list)
