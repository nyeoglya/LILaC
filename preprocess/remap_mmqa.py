import os
import json
import urllib.parse
import typing as tp
from utils_mmqa import mmqa_get_clean_wikidocs_titles

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
    print(mmqa_find_component_from_file("/dataset/original/mmqa/"))
