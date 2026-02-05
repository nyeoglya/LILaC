from __future__ import annotations
import threading

from PIL import Image
import cv2

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
            new_item = {}
            if gen_input.text:
                new_item["txt"] = gen_input.text
            if gen_input.img_path and gen_input.img_path.lower().endswith('.png'):
                if gen_input.bounding_box:
                    original_image = cv2.imread(gen_input.img_path)
                    h, w = original_image.shape[:2]
                    x1, y1, x2, y2 = map(int, gen_input.bounding_box)
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError("Invalid bounding box")
                    new_item["img"] = cv2.cvtColor(original_image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                else:
                    new_item["img"] = Image.open(gen_input.img_path).convert("RGB")
            
            outputs = self.model.encode(
                [new_item],
                is_query=False,
                max_length=512,
            )

            return outputs["hidden_states"][0].cpu()
    
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
