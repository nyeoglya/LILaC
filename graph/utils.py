import requests
import numpy as np

import typing as tp
from dataclasses import dataclass

MMEMBED_SERVER_URL = "http://lilac-mmembed:8002"

@dataclass
class EmbeddingRequestData:
    instruction: str = ""
    text: str = ""
    img_path: str = ""

def get_embedding(reqeust_data: EmbeddingRequestData) -> tp.List[float]:
    payload = {
        "instruction": reqeust_data.instruction,
        "text": reqeust_data.text,
        "img_path": reqeust_data.img_path,
    }

    r = requests.post(f"{MMEMBED_SERVER_URL}/embed", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)
    return embedding

def get_batch_embedding(request_data_list: tp.List[EmbeddingRequestData]) -> tp.List[float]:
    payloads = [{
        "instruction": request_data.instruction,
        "text": request_data.text,
        "img_path": request_data.img_path,
    } for request_data in request_data_list]

    r = requests.post(f"{MMEMBED_SERVER_URL}/embed/batch", json={"items": payloads}, timeout=120)
    r.raise_for_status()
    
    data_list = r.json()
    vectors = data_list["embeddings"]
    embeddings = [np.array(vec, dtype=np.float32) for vec in vectors]
    return embeddings
