from __future__ import annotations

from dataclasses import dataclass

@dataclass
class GenerationInput:
    instruction: str = ""
    text: str = ""
    img_path: str = ""
