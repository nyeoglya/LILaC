from __future__ import annotations
import typing as tp

from PIL import Image

import torch
from transformers import AutoModel

from utils import GenerationInput

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
        
    @torch.inference_mode()
    def get_embeddings(self, gen_inputs: tp.List[GenerationInput]):
        items = []

        for gi in gen_inputs:
            item = {}
            if gi.text:
                item["txt"] = gi.text
            if gi.img_paths:
                item["img"] = [Image.open(p).convert("RGB") for p in gi.img_paths]
            items.append(item)

        outputs = self.model.encode(
            items,
            is_query=True,
            instruction=gen_inputs[0].instruction,
            max_length=4096,
        )

        return outputs["hidden_states"].cpu()
