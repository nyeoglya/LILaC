import os
import pickle
import typing as tp

import cv2

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

class BatchImageEmbedder:
    def __init__(
        self,
        image_file_list: tp.List[str],
        device: str = "cuda:0",
        image_size: int = 512,
    ):
        assert torch.cuda.is_available()
        
        MODEL_NAME: str = "facebook/dinov2-base"
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME,
            size={"shortest_edge": image_size},
            do_center_crop=True,
        )
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        self.model.eval()
    
        self.image_filepath_list: tp.List[str] = image_file_list
        self.progress_bar = tqdm(total=0, desc="Batch Image Embedding...")

    def run_embedding(
        self,
        failed_filepath: str,
        result_embedding_filepath: str,
        batch_size: int = 16,
    ):
        self.progress_bar.total = len(self.image_filepath_list)
        
        result_embedding_map = {} 
        failed_file_list: tp.List[str] = []
        for i in range(0, len(self.image_filepath_list), batch_size):
            batch_image_filepath_list: tp.List[str] = self.image_filepath_list[i : i + batch_size]

            image_batch_input_list: tp.List[cv2.typing.MatLike] = []
            valid_image_filepath_list: tp.List[str] = []
            for image_filepath in batch_image_filepath_list:
                image = cv2.imread(image_filepath)
                if image is None:
                    self.progress_bar.update(1)
                    failed_file_list.append(image_filepath)
                    continue
                image_batch_input_list.append(image)
                valid_image_filepath_list.append(image_filepath)
            if not image_batch_input_list:
                continue

            image_batch_output_list: tp.List[np.ndarray] = self.batch_embed(image_batch_input_list)
            self.progress_bar.update(len(valid_image_filepath_list))

            for path, emb in zip(valid_image_filepath_list, image_batch_output_list):
                result_embedding_map[path] = emb
        
        with open(result_embedding_filepath, "wb") as f:
            pickle.dump(result_embedding_map, f)
        
        with open(failed_filepath, "w", encoding="utf-8") as failed_file:
            for path in failed_file_list:
                failed_file.write(path + "\n")

    @torch.inference_mode()
    def batch_embed(
        self,
        image_input_list: tp.List[np.ndarray],
    ) -> tp.List[np.ndarray]:
        rgb_image_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_input_list]

        input_list = self.processor(
            images=rgb_image_list,
            return_tensors="pt",
        ).to(self.device)

        output_list = self.model(**input_list)

        result_embeding_list = output_list.last_hidden_state[:, 0, :]
        result_embeding_list = F.normalize(result_embeding_list.float(), dim=-1)

        return result_embeding_list.detach().cpu().numpy()