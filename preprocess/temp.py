import os
from pathlib import Path

from utils import get_clean_savepath
from utils_mmqa import mmqa_get_clean_wikidocs_titles
from preprocess import MMQA_PATH, MMQA_CRAWL_IMAGE_FOLDER, MMQA_PARSE_JSON_FOLDER
from crawler.wiki import BatchWikiImageCrawler


def get_actual_pending_downloads(target_url_list: list, save_folderpath: str):
    """
    1. 로컬 파일 목록을 읽어온다.
    2. 로컬 파일명을 가상으로 clean 규칙에 맞춰 변환해본다.
    3. 새로 다운로드할 URL들을 clean 규칙에 맞춘 이름과 대조한다.
    4. 이미 있는 녀석들을 제외한 '진짜 신규' 리스트를 반환한다.
    """
    
    # 1. 로컬에 있는 실제 파일명들을 내 규칙(clean)으로 가상 변환한 집합 생성
    # 실제 파일은 이름이 안 바뀌고, 메모리 안에서만 바뀐 이름으로 인식됨
    virtual_existing_names = set()
    
    actual_files = [f for f in os.listdir(save_folderpath) if os.path.isfile(os.path.join(save_folderpath, f))]
    
    for real_name in actual_files:
        stem = Path(real_name).stem
        ext = Path(real_name).suffix.lstrip('.')
        
        # 가상으로 경로를 생성한 뒤 이름만 추출
        v_path = get_clean_savepath(save_folderpath, stem, ext)
        v_name = os.path.basename(v_path)
        virtual_existing_names.add(v_name)

    # 2. 다운로드 대상 URL들 중 없는 것만 골라내기
    pending_urls = []
    
    target_expected_names = set()
    for url in target_url_list:
        original_fn = url.split('/')[-1]
        name_part, ext_part = os.path.splitext(original_fn)
        target_path = get_clean_savepath(save_folderpath, name_part, ext_part.lstrip('.'))
        target_expected_names.add(os.path.basename(target_path))

    # 2. 로컬 파일들(virtual_existing_names)을 하나씩 검사
    # 로컬에는 있지만, 현재 타겟 목록에는 없는 파일들을 추출
    orphaned_files = [] # 현재 목록에 없는 로컬 파일들

    # virtual_existing_names는 이전 로직에서 만든 '로컬 파일의 clean 명칭 리스트'입니다.
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

# Image Crawler
batch_image_crawler = BatchWikiImageCrawler(MMQA_CRAWL_IMAGE_FOLDER)
batch_image_crawler.set_clean_imglinks_from_folder(MMQA_PARSE_JSON_FOLDER)

get_actual_pending_downloads(batch_image_crawler.image_data_url_list, "/dataset/backup/mmqa_image")
