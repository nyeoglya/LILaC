import typing as tp
import os
import requests

import urllib.parse

QWEN_SERVER_URL = "http://lilac-qwen:8003"

def get_clean_savepath_from_url(image_save_folder_path: str, original_url: str) -> str:
    url_path: str = original_url.split('?')[0]
    raw_filename: str = url_path.split('/')[-1]
    decoded_name: str = urllib.parse.unquote(raw_filename)
    name_without_ext, extension = os.path.splitext(decoded_name)
    extension = extension.replace(".","")
    if not extension:
        extension: str = "jpg"
    return get_clean_savepath(image_save_folder_path, name_without_ext, extension)

def get_clean_savepath(save_folderpath: str, filename: str, extension: str = "") -> str:
    clean_file_name: str = "".join([c for c in filename if c.isalnum() or c in (' ', '_', '-')]).rstrip()
    clean_file_path: str = os.path.join(save_folderpath, f"{clean_file_name}.{extension}")
    return clean_file_path

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

def get_llm_response(text: str, imgpath_list: tp.List[str], server_url: str):
    payload = {
        "instruction": "",
        "text": text,
        "img_paths": imgpath_list,
    }

    r = requests.post(f"{server_url}/generate", json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    return data["response"]
