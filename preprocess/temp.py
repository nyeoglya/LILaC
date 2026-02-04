import os
from pathlib import Path

from utils import get_clean_savepath
from utils_mmqa import mmqa_get_clean_wikidocs_titles
from preprocess import MMQA_PATH, MMQA_CRAWL_IMAGE_FOLDER, MMQA_PARSE_JSON_FOLDER
from crawler.wiki import BatchWikiImageCrawler


def get_actual_pending_downloads(target_url_list: list, save_folderpath: str):
    virtual_existing_names = set()
    
    actual_files = [f for f in os.listdir(save_folderpath) if os.path.isfile(os.path.join(save_folderpath, f))]
    
    for real_name in actual_files:
        stem = Path(real_name).stem
        ext = Path(real_name).suffix.lstrip('.')
        
        v_path = get_clean_savepath(save_folderpath, stem, ext)
        v_name = os.path.basename(v_path)
        virtual_existing_names.add(v_name)

    pending_urls = []
    
    target_expected_names = set()
    for url in target_url_list:
        original_fn = url.split('/')[-1]
        name_part, ext_part = os.path.splitext(original_fn)
        target_path = get_clean_savepath(save_folderpath, name_part, ext_part.lstrip('.'))
        target_expected_names.add(os.path.basename(target_path))

    orphaned_files = []

    for v_name in virtual_existing_names:
        if v_name not in target_expected_names:
            orphaned_files.append(v_name)

    print(f"📊 역방향 대조 결과")
    print(f"- 현재 타겟(JSON) 이미지: {len(target_expected_names)}개")
    print(f"- 백업 폴더 내 유니크 파일: {len(virtual_existing_names)}개")
    print(f"- 타겟 목록에 없는 잉여 파일: {len(orphaned_files)}개")
    print(orphaned_files[:10])
    
    return pending_urls

mmqa_wiki_doc_title_list = mmqa_get_clean_wikidocs_titles(MMQA_PATH)

batch_image_crawler = BatchWikiImageCrawler(MMQA_CRAWL_IMAGE_FOLDER)
batch_image_crawler.set_clean_imglinks_from_folder(MMQA_PARSE_JSON_FOLDER)

get_actual_pending_downloads(batch_image_crawler.image_data_url_list, "/dataset/backup/mmqa_image")
