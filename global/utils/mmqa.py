import os
import json
import typing as tp
import urllib.parse

def mmqa_get_clean_wikidocs_titles(mmqa_path: str) -> tp.List[str]:
    mmqa_devpath = os.path.join(mmqa_path, "MMQA_dev.jsonl")
    mmqa_textpath = os.path.join(mmqa_path, "MMQA_texts.jsonl")
    mmqa_imagepath = os.path.join(mmqa_path, "MMQA_images.jsonl")
    mmqa_tablepath = os.path.join(mmqa_path, "MMQA_tables.jsonl")
    
    doc_ids = set()
    img_ids = set()
    tab_ids = set()

    with open(mmqa_devpath, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            for ctx in data.get("supporting_context", []):
                d_id = ctx.get('doc_id')
                d_part = ctx.get('doc_part')
                if not d_id: continue
                    
                if d_part == "text": doc_ids.add(d_id)
                elif d_part == "image": img_ids.add(d_id)
                elif d_part == "table": tab_ids.add(d_id)
        
    final_links = set()

    def map_ids_to_urls(filepath, target_ids):
        found_urls = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # target_ids에 있는지 확인 (id가 정확히 일치해야 함)
                if item.get('id') in target_ids:
                    found_urls.append(item.get('url', ''))
        return found_urls

    final_links.update(map_ids_to_urls(mmqa_textpath, doc_ids))
    final_links.update(map_ids_to_urls(mmqa_imagepath, img_ids))
    final_links.update(map_ids_to_urls(mmqa_tablepath, tab_ids))

    clean_titles = set()
    for url in final_links:
        if not url: continue
        decoded_url = urllib.parse.unquote(url)
        title = decoded_url.replace('https://en.wikipedia.org/wiki/', '').replace(' ', '_')
        clean_titles.add(title)
            
    print(f"Extract {len(clean_titles)} unique wiki title")
    return list(clean_titles)
