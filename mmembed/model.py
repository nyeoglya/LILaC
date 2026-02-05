from __future__ import annotations
import threading

from PIL import Image

import torch
from transformers import AutoModel

from utils import GenerationInput, QueryGenerationInput

class MMEmbed:
    def __init__(self, device: str):
        self.device = torch.device(device)
        print(f"[MMEmbed] Using device: {self.device}")
        print("[MMEmbed] Loading model...")
        self.model = AutoModel.from_pretrained(
            "nvidia/MM-Embed",
            trust_remote_code=True,
        ).to(self.device).eval()
        print("[MMEmbed] Model loaded successfully.")
        self._lock = threading.Lock()
    
    @torch.inference_mode()
    def embedding(self, gen_input: GenerationInput) -> torch.Tensor:
        with self._lock:
            item = {}
            if gen_input.text:
                item["txt"] = gen_input.text
            if gen_input.img_path and gen_input.img_path.lower().endswith('.png'):
                item["img"] = Image.open(gen_input.img_path).convert("RGB")
            
            outputs = self.model.encode(
                [item],
                is_query=False,
                max_length=512,
            )

            return outputs["hidden_states"].cpu()
    
    @torch.inference_mode()
    def query_embedding(self, query_gen_input: QueryGenerationInput) -> torch.Tensor:
        with self._lock:
            item = {}
            if query_gen_input.text:
                item["txt"] = query_gen_input.text
            if query_gen_input.img_path and query_gen_input.img_path.lower().endswith('.png'):
                item["img"] = Image.open(query_gen_input.img_path).convert("RGB")
            
            outputs = self.model.encode(
                [item],
                is_query=True,
                instruction=query_gen_input.instruction,
                max_length=512,
            )

            return outputs["hidden_states"].cpu()
