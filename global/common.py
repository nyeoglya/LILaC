import os
import requests
import urllib.parse
import typing as tp
from dataclasses import dataclass

import numpy as np

def get_clean_savepath_from_url_with_custom_extension(image_save_folder_path: str, original_url: str, extension: str) -> str:
    name_without_ext, _ = get_clean_filename_and_extension_from_url(original_url)
    return get_clean_savepath(image_save_folder_path, name_without_ext, extension)

def get_clean_savepath_from_url(image_save_folder_path: str, original_url: str) -> str:
    name_without_ext, extension = get_clean_filename_and_extension_from_url(original_url)
    extension = extension.replace(".","")
    if not extension:
        extension: str = "png"
    return get_clean_savepath(image_save_folder_path, name_without_ext, extension)

def get_clean_filename_from_url(original_url: str) -> str:
    url_path: str = original_url.split('?')[0]
    raw_filename: str = url_path.split('/')[-1]
    decoded_name: str = urllib.parse.unquote(raw_filename)
    name_without_ext, _ = os.path.splitext(decoded_name)
    return get_clean_filename(name_without_ext)

def get_clean_filename_and_extension_from_url(original_url: str) -> tp.Tuple[str, str]:
    url_path: str = original_url.split('?')[0]
    raw_filename: str = url_path.split('/')[-1]
    decoded_name: str = urllib.parse.unquote(raw_filename)
    name_without_ext, extension = os.path.splitext(decoded_name)
    return get_clean_filename(name_without_ext), extension

def get_clean_filename_with_extension_from_filepath(original_path: str) -> str:
    raw_filename: str = original_path.split('/')[-1]
    name_without_ext, extension = os.path.splitext(raw_filename)
    return get_clean_filename(name_without_ext) + extension

def get_clean_savepath(save_folderpath: str, filename: str, extension: str = "") -> str:
    clean_file_name: str = get_clean_filename(filename)
    clean_file_path: str = os.path.join(save_folderpath, f"{clean_file_name}.{extension}")
    return clean_file_path

def get_clean_filename(filename: str) -> str:
    clean_file_name: str = "".join([c for c in filename if c.isalnum() or c in (' ', '_', '-')]).rstrip()
    return clean_file_name

def save_html_content_to_file(doc_html_clean_filepath: str, html_content: str):
    try:
        with open(doc_html_clean_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return True
    except Exception as e:
        print(f"Failed to save {doc_html_clean_filepath}: {e}")
        return False

def save_image_to_file(image_clean_filepath: str, image_content: bytes):
    try:
        with open(image_clean_filepath, "wb") as f:
            f.write(image_content)
        return True
    except Exception as e:
        print(f"Failed to save {image_clean_filepath}: {e}")
        return False


@dataclass
class EmbeddingRequestData:
    text: str = ""
    img_path: str = ""
    bounding_box: tp.Optional[tp.Tuple[int, int, int, int]] = None

def get_query_embedding(server_url: str, instruction: str, text: str, img_path="") -> np.ndarray:
    payload = {
        "instruction": instruction,
        "text": text,
        "img_path": img_path,
    }

    r = requests.post(f"{server_url}/embed/query", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)
    return embedding

def get_embedding(server_url: str, reqeust_data: EmbeddingRequestData) -> np.ndarray:
    payload = {
        "text": reqeust_data.text,
        "img_path": reqeust_data.img_path,
        "bounding_box": reqeust_data.bounding_box,
    }

    r = requests.post(f"{server_url}/embed", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)
    return embedding

def get_batch_embedding(server_url: str, request_data_list: tp.List[EmbeddingRequestData]) -> tp.List[np.ndarray]:
    payloads = [{
        "text": request_data.text,
        "img_path": request_data.img_path,
    } for request_data in request_data_list]

    r = requests.post(f"{server_url}/embed/batch", json={"items": payloads}, timeout=120)
    r.raise_for_status()
    
    data_list = r.json()
    vectors = data_list["embeddings"]
    embeddings = [np.array(vec, dtype=np.float32) for vec in vectors]
    return embeddings

def get_llm_response(server_url: str, text: str, image_filepath_list: tp.List[str] = [], max_tokens: int = 256):
    payload = {
        "text": text,
        "img_paths": image_filepath_list,
        "max_tokens": max_tokens
    }

    r = requests.post(f"{server_url}/generate", json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    return data["response"]
