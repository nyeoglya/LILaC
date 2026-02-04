import os
import requests
import typing as tp
import urllib.parse
from dataclasses import dataclass

import numpy as np

PARSED_JSON_FOLDER = "/dataset/parse/mmqa_json/"
IMG_FOLDER = "/dataset/crawl/mmqa_image/"
LDOC_FOLDER = "/dataset/process/mmqa/"

GRAPH_TEMP_FILE = "lilac_temp.jsonl"
LLM_TEMP_FILE = "lilac_llm_temp.jsonl"
FINAL_RESULT_FILENAME = "lilac_query_answers.json"

MMEMBED_SERVER_URL = "http://lilac-mmembed:8002"
QWEN_SERVER_URL = "http://lilac-qwen:8003"

@dataclass
class EmbeddingRequestData:
    text: str = ""
    img_path: str = ""

def get_clean_savepath_from_url(image_save_folder_path: str, original_url: str) -> str:
    url_path: str = original_url.split('?')[0]
    raw_filename: str = url_path.split('/')[-1]
    decoded_name: str = urllib.parse.unquote(raw_filename)
    name_without_ext, extension = os.path.splitext(decoded_name)
    extension = extension.replace(".","")
    if not extension:
        extension: str = "jpg"
    return get_clean_savepath(image_save_folder_path, name_without_ext, extension)

def get_clean_savepath(save_folderpath: str, filename: str, extension: str) -> str:
    clean_file_name: str = "".join([c for c in filename if c.isalnum() or c in (' ', '_', '-')]).rstrip()
    clean_file_path: str = os.path.join(save_folderpath, f"{clean_file_name}.{extension}")
    return clean_file_path

def get_query_embedding(instruction, text, img_path="") -> np.ndarray:
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

def get_embedding(reqeust_data: EmbeddingRequestData) -> np.ndarray:
    payload = {
        "text": reqeust_data.text,
        "img_path": reqeust_data.img_path,
    }

    r = requests.post(f"{MMEMBED_SERVER_URL}/embed", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)
    return embedding

def get_batch_embedding(request_data_list: tp.List[EmbeddingRequestData]) -> tp.List[np.ndarray]:
    return [] # TODO: 변경하기

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
