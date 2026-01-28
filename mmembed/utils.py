from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Dict

TIME_MULTIPLIER_MS = 1000.0

@dataclass
class GenerationInput:
    text_prompt: str | None  = None
    system_prompt: Optional[str] = None
    image_paths: List[str] | None  = None

@dataclass
class GenerationParameters:
    show_progress: bool = False
    batch_size: int  = 1

@dataclass
class APIUsage:
    model: str = ""
    rate_info: Dict[str, float] | None  = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent = 4)
    
    def __iadd__(self, other: APIUsage) -> APIUsage:
        if other is None:
            return self
        
        if other.input_tokens is not None:
            self.input_tokens += other.input_tokens
        
        if other.output_tokens is not None:
            self.output_tokens += other.output_tokens
        
        if other.cached_input_tokens is not None:
            self.cached_input_tokens += other.cached_input_tokens
        
        if other.estimated_cost_usd is not None:
            self.estimated_cost_usd += other.estimated_cost_usd
        
        return self

    def __add__(self, other: APIUsage) -> APIUsage:
        if other is None:
            return APIUsage(
                model               = self.model,
                rate_info           = self.rate_info,
                input_tokens        = self.input_tokens,
                output_tokens       = self.output_tokens,
                cached_input_tokens = self.cached_input_tokens,
                estimated_cost_usd  = self.estimated_cost_usd,
            )

        aggregated = APIUsage(
            model               = self.model or other.model,
            rate_info           = self.rate_info or other.rate_info,
            input_tokens        = self.input_tokens,
            output_tokens       = self.output_tokens,
            cached_input_tokens = self.cached_input_tokens,
            estimated_cost_usd  = self.estimated_cost_usd,
        )
        aggregated += other
        return aggregated

    def to_dict(self):
        return {
            "model":                 self.model,
            "rate_info":             self.rate_info,
            "input_tokens":          self.input_tokens,
            "output_tokens":         self.output_tokens,
            "cached_input_tokens":   self.cached_input_tokens,
            "estimated_cost_usd":    self.estimated_cost_usd,
        }

@dataclass
class GenerationOutput:
    text: str

@dataclass
class GenMetadata:
    time: float | None = None
    time_preprocess: float | None = None
    time_infer: float | None = None
    time_postprocess: float | None = None
    num_inputs: int | None = None
    batch_size: int | None = None
    batch_iterations: int | None = None
    api_usage: APIUsage | None = None

    def __str__(self) -> str:
        data = {}
        for key, value in self.__dict__.items():
            if value is None:
                data[key] = None
            elif isinstance(value, APIUsage):
                data[key] = value.__dict__
            else:
                data[key] = value
        return json.dumps(data, indent = 4)

    def __iadd__(self, other: GenMetadata) -> GenMetadata:
        if other.time is not None:
            if self.time is None:
                self.time = 0.0
            self.time += other.time
        
        if other.time_preprocess is not None:
            if self.time_preprocess is None:
                self.time_preprocess = 0.0
            self.time_preprocess += other.time_preprocess
        
        if other.time_infer is not None:
            if self.time_infer is None:
                self.time_infer = 0.0
            self.time_infer += other.time_infer
        
        if other.time_postprocess is not None:
            if self.time_postprocess is None:
                self.time_postprocess = 0.0
            self.time_postprocess += other.time_postprocess

        if other.api_usage is not None:
            if self.api_usage is None:
                self.api_usage = APIUsage()
            self.api_usage += other.api_usage
        
        return self
