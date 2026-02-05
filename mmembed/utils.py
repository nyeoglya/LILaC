from __future__ import annotations
import typing as tp

from dataclasses import dataclass

@dataclass
class GenerationInput:
    text: str = ""
    img_path: str = ""
    bounding_box: tp.Optional[tp.Tuple[int, int, int, int]] = None

@dataclass
class QueryGenerationInput:
    instruction: str = ""
    text: str = ""
    img_path: str = ""
