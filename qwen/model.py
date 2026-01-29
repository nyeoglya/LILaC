from __future__ import annotations
import typing as tp

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils import GenerationInput, GenerationOutput

class Qwen3_VL:
    def __init__(self, device: str):
        print(f"[Qwen3_VL] Using device: {device}")
        print(f"[Qwen3_VL] Loading model...")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        ).eval()
        print("[Qwen3_VL] Model and tokenizer loaded successfully.")

    @torch.inference_mode()
    def inference(self, gen_input: GenerationInput, max_tokens: int) -> GenerationOutput:
        return self.batch_inference([gen_input], batch_size=1, max_tokens=max_tokens)[0]

    @torch.inference_mode()
    def batch_inference(self, gen_inputs: tp.List[GenerationInput], batch_size: int, max_tokens: int) -> tp.List[GenerationOutput]:
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.0,
        )

        gen_outputs: tp.List[GenerationOutput] = []
        batch_iter = range(0, len(gen_inputs), batch_size)

        for start in batch_iter:
            batch_inputs = gen_inputs[start : start + batch_size]

            # build messages
            batch_messages = []
            for inp in batch_inputs:
                user_content = [{"type": "image", "image": img_path} for img_path in inp.img_paths]
                user_content.append({"type": "text", "text": inp.text})
                messages = [
                    {"role": "system", "content": inp.instruction},
                    {"role": "user", "content": user_content},
                ]
                batch_messages.append(messages)

            # processor handles everything
            inputs = self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            # generate
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

            # trim prompt & decode
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
                reply_ids = out_ids[len(in_ids):]
                text = self.processor.decode(
                    reply_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()

                gen_outputs.append(GenerationOutput(text=text))

        return gen_outputs
