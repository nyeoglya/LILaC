from __future__ import annotations

from dataclasses import dataclass

@dataclass
class GenerationInput:
    text: str = ""
    img_path: str = ""

@dataclass
class QueryGenerationInput:
    instruction: str = ""
    text: str = ""
    img_path: str = ""
