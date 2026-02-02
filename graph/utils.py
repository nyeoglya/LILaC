import os
import requests
import typing as tp
from urllib.parse import unquote
from dataclasses import dataclass

import numpy as np

JSON_FOLDER = "/dataset/crawl/mmqa_html/"
IMG_FOLDER = "/dataset/crawl/mmqa_image/"
LDOC_FOLDER = "/dataset/process/mmqa/"
FINAL_RESULT_FILENAME = "lilac_query_answers.json"

MMEMBED_SERVER_URL = "http://lilac-mmembed:8002"
QWEN_SERVER_URL = "http://lilac-qwen:8003"

@dataclass
class EmbeddingRequestData:
    text: str = ""
    img_path: str = ""

def get_clean_imagepath(img_folder, image_str):
    invalid_chars = '<>:"/\\|?*'
    clean_name = unquote(image_str)
    if "File:" in clean_name:
        clean_name = clean_name.split("File:")[1]
    elif "https://" in clean_name:
        clean_name = clean_name.split("/")[-1]
    for char in invalid_chars:
        clean_name = clean_name.replace(char, '')

    full_path = os.path.join(img_folder, clean_name)
    return full_path

def get_query_embedding(instruction, text, img_path="") -> tp.List[float]:
    payload = {
        "instruction": instruction,
        "text": text,
        "img_path": img_path,
    }

    r = requests.post(f"{MMEMBED_SERVER_URL}/embed/query", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)
    return embedding

def get_embedding(reqeust_data: EmbeddingRequestData) -> tp.List[float]:
    payload = {
        "text": reqeust_data.text,
        "img_path": reqeust_data.img_path,
    }

    r = requests.post(f"{MMEMBED_SERVER_URL}/embed", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)
    return embedding

def get_batch_embedding(request_data_list: tp.List[EmbeddingRequestData]) -> tp.List[np.array]:
    payloads = [{
        "text": request_data.text,
        "img_path": request_data.img_path,
    } for request_data in request_data_list]

    r = requests.post(f"{MMEMBED_SERVER_URL}/embed/batch", json={"items": payloads}, timeout=120)
    r.raise_for_status()
    
    data_list = r.json()
    vectors = data_list["embeddings"]
    embeddings = [np.array(vec, dtype=np.float32) for vec in vectors]
    return embeddings

def get_llm_response(instruction, text, img_paths=[]):
    payload = {
        "instruction": instruction,
        "text": text,
        "img_paths": img_paths,
    }

    r = requests.post(f"{QWEN_SERVER_URL}/generate", json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    return data["response"]
