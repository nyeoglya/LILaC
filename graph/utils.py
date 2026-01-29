import requests
import numpy as np

import typing as tp

MMEMBED_SERVER_URL = "http://lilac-mmembed:8002"

def query_llm(data):
    pass

def get_embedding(instruction, text, img_paths) -> tp.List[float]:
    payload = {
        "instruction": instruction,
        "text": text,
        "img_paths": img_paths,
    }

    r = requests.post(f"{MMEMBED_SERVER_URL}/embed", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)
    return embedding
