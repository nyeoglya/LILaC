import os

import urllib.parse

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

def convert_image_to_png(image_folderpath: str) -> bool:
    # TODO: size 고정
    # TODO: 확장자 변경    
    return False
