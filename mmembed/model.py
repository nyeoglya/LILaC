from __future__ import annotations
import io
import typing as tp

from PIL import Image
import cairosvg

import torch
from transformers import AutoModel

from utils import GenerationInput, QueryGenerationInput

def load_svg(svg_path):
    png_data = cairosvg.svg2png(url=svg_path)
    if png_data:
        image = Image.open(io.BytesIO(png_data))
        return image
    else:
        return None

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
    def embedding(self, gen_input: GenerationInput) -> torch.Tensor:
        item = {}
        if gen_input.text:
            item["txt"] = gen_input.text
        if gen_input.img_path:
            if gen_input.img_path.lower().endswith('.svg'):
                loaded_svg_img = load_svg(gen_input.img_path)
                if loaded_svg_img:
                    item["img"] = loaded_svg_img.convert("RGB")
            else:
                item["img"] = Image.open(gen_input.img_path).convert("RGB")
        
        outputs = self.model.encode(
            [item],
            is_query=False,
            max_length=512,
        )

        return outputs["hidden_states"].cpu()
    
    @torch.inference_mode()
    def query_embedding(self, query_gen_input: QueryGenerationInput) -> torch.Tensor:
        item = {}
        if query_gen_input.text:
            item["txt"] = query_gen_input.text
        if query_gen_input.img_path:
            if query_gen_input.img_path.lower().endswith('.svg'):
                loaded_svg_img = load_svg(query_gen_input.img_path)
                if loaded_svg_img:
                    item["img"] = loaded_svg_img.convert("RGB")
            else:
                item["img"] = Image.open(query_gen_input.img_path).convert("RGB")
        
        outputs = self.model.encode(
            [item],
            is_query=True,
            instruction=query_gen_input.instruction,
            max_length=512,
        )

        return outputs["hidden_states"].cpu()

    @torch.inference_mode()
    def batch_embedding(self, gen_inputs: tp.List[GenerationInput]):
        items: tp.List[dict] = []

        # GenerationInput -> item dict 변환
        for gi in gen_inputs:
            item = {}

            item["txt"] = gi.text or ""

            if gi.img_path:
                if gi.img_path.lower().endswith(".svg"):
                    loaded_svg_img = load_svg(gi.img_path)
                    if loaded_svg_img:
                        item["img"] = loaded_svg_img.convert("RGB")
                else:
                    with Image.open(gi.img_path) as img:
                        item["img"] = img.convert("RGB")

            items.append(item)

        # index 유지
        text_only = []
        text_image = []

        for idx, item in enumerate(items):
            if "img" in item:
                text_image.append((idx, item))
            else:
                text_only.append((idx, item))

        # 결과 버퍼
        embeddings = [None] * len(items)

        # text-only batch encode
        if text_only:
            text_batches = [text_only[i:i+2] for i in range(0, len(text_only), 2)]
            idx_list = []
            batch_list = []
            for text_batch in text_batches:
                idxs, batch_items = zip(*text_batch)
                embs = self.model.encode(
                    list(batch_items),
                    is_query=False,
                    max_length=512,
                )["hidden_states"]
                idx_list.extend(idxs)
                batch_list.extend(embs)

            for i, emb in zip(idx_list, batch_list):
                embeddings[i] = emb

        # text+image batch encode
        if text_image:
            image_batches = [text_image[i:i+2] for i in range(0, len(text_image), 2)]
            idx_list = []
            batch_list = []
            for image_batch in image_batches:
                idxs, batch_items = zip(*image_batch)
                embs = self.model.encode(
                    list(batch_items),
                    is_query=False,
                    max_length=512,
                )["hidden_states"]
                idx_list.extend(idxs)
                batch_list.extend(embs)

            for i, emb in zip(idx_list, batch_list):
                embeddings[i] = emb

        torch.cuda.empty_cache()
        # 최종 stack
        return torch.stack(embeddings, dim=0).cpu()
