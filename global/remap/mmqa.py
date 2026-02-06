import os
import json
import urllib.parse
import typing as tp

from utils_mmqa import mmqa_get_clean_wikidocs_titles
from image_descriptor import BatchImageRemapEmbedder
from config import (
    MMQA_PATH,
    MMQA_IMAGE_REFERENCE_PATH,
    MMQA_REMAP_IMAGE_EMBEDDING_FAILED_FILEPATH,
    MMQA_REMAP_REFERENCE_EMBEDDING_FAILED_FILEPATH,
    MMQA_REMAP_IMAGE_EMBEDDING_PT,
    MMQA_REMAP_REFERENCE_EMBEDDING_PT
)

def mmqa_find_component_from_file(mmqa_path: str) -> tp.Dict[str, tp.Dict[str, tp.List[str]]]:
    clean_titles: tp.List[str] = mmqa_get_clean_wikidocs_titles(mmqa_path)
    mmqa_textpath = os.path.join(mmqa_path, "MMQA_texts.jsonl")
    mmqa_imagepath = os.path.join(mmqa_path, "MMQA_images.jsonl")
    mmqa_tablepath = os.path.join(mmqa_path, "MMQA_tables.jsonl")

    result = {
        title: {"txtid": [], "imgid": [], "tabid": []}
        for title in clean_titles
    }

    def scan_component(filepath: str, key: str):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                url = item.get("url")
                cid = item.get("id")
                if not url or not cid:
                    continue

                decoded = urllib.parse.unquote(url)
                title = (
                    decoded
                    .replace("https://en.wikipedia.org/wiki/", "")
                    .replace(" ", "_")
                )

                if title in result:
                    result[title][key].append(cid)

    scan_component(mmqa_textpath, "txtid")
    scan_component(mmqa_imagepath, "imgid")
    scan_component(mmqa_tablepath, "tabid")

    print(f"Extract {len(result)} wiki titles with components")
    return result

if __name__ == "__main__":
    mmqa_component_map = mmqa_find_component_from_file(MMQA_PATH)
    processed_imagepath_list = [datum for data in mmqa_component_map.values() for datum in data["imgid"]]
    mmqa_fullimage_map = {}
    for fname in os.listdir(MMQA_IMAGE_REFERENCE_PATH):
        stem, ext = os.path.splitext(fname)
        if ext:
            mmqa_fullimage_map[stem] = ext.lstrip(".")

    mmqa_reference_image_list = []
    for data in processed_imagepath_list:
        if data in mmqa_fullimage_map:
            mmqa_reference_image_list.append(os.path.join(MMQA_IMAGE_REFERENCE_PATH, "f{data}.{mmqa_fullimage_map[data]}"))

    embedder = BatchImageRemapEmbedder(mmqa_reference_image_list)
    embedder.run_embedding(MMQA_REMAP_REFERENCE_EMBEDDING_FAILED_FILEPATH, MMQA_REMAP_REFERENCE_EMBEDDING_PT)
    
    processed_imagepath_list = ["/dataset/process/mmqa_image/" + filename for filename in os.listdir("/dataset/process/mmqa_image")]
    embedder = BatchImageRemapEmbedder(processed_imagepath_list)
    embedder.run_embedding(MMQA_REMAP_IMAGE_EMBEDDING_FAILED_FILEPATH, MMQA_REMAP_IMAGE_EMBEDDING_PT)
