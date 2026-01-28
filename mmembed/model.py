from __future__ import annotations
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from transformers import logging as tf_logging

from utils import GenerationInput

tf_logging.set_verbosity_error()


class MMEmbed:
    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = True,
    ):
        self.model_name = "mm-embed"
        self.model_path = model_path

        print("[MMEmbed] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        print("[MMEmbed] Loading model (multi-GPU shard)...")
        self.model = AutoModel.from_pretrained(
            model_path,
            device_map=device_map,
            token=True,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",  # Titan Xp / 안정성 우선
            low_cpu_mem_usage=True,
        ).eval()

        print("[MMEmbed] Model loaded successfully.")

    @torch.inference_mode()
    def get_embeddings(
        self,
        gen_inputs: List[GenerationInput],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Returns:
            Tensor [N, D] on CPU
        """
        all_embeddings = []

        for i in range(0, len(gen_inputs), batch_size):
            batch = gen_inputs[i : i + batch_size]

            # ⚠️ device 이동 금지 (accelerate가 자동 처리)
            inputs = self.processor(
                text=[inp.text_prompt for inp in batch],
                padding=True,
                return_tensors="pt",
            )

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # mm-embed / LlavaNext 구조 대응
            if not hasattr(outputs, "language_model_outputs"):
                raise RuntimeError("Unexpected model output structure")

            hidden_states = outputs.language_model_outputs.hidden_states
            last_hidden = hidden_states[-1]  # [B, T, D]

            attention_mask = inputs.attention_mask.to(last_hidden.device)

            # ✅ last-token pooling (가장 안정적)
            last_token_index = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)

            embeddings = last_hidden[batch_indices, last_token_index]

            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
