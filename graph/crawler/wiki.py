import os
import re
import json
import time
import random
import typing as tp

import threading
from concurrent.futures import ThreadPoolExecutor

import urllib.parse
import requests

from base import *

class WikiBatchCrawler:
    def __init__(self, folder_path="crawled_html"):
        self.base_url = f"https://en.wikipedia.org/api/rest_v1/page/html/"
        self.headers = {
            "User-Agent": "LILaCBulkScraper/1.0 (abc@example.com)"
        }
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
        
        self.folder_path = folder_path
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.progress = 0
        self.max_progress = 0
        self.lock = threading.Lock()

    def get_filepath(self, page_name):
        safe_filename = "".join([c for c in page_name if c.isalnum() or c in (' ', '_', '-')]).rstrip()
        file_path = os.path.join(self.folder_path, f"{safe_filename}.html")
        return file_path

    def save_to_file(self, page_name, html_content):
        file_path = self.get_filepath(page_name)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return True
        except Exception as e:
            print(f"Failed to save {page_name}: {e}")
            return False

    def fetch_and_save(self, page_name):
        url = f"{self.base_url.rstrip('/')}/{page_name}"
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            with self.lock:
                self.progress += 1
                if self.progress % 50 == 0:
                    print(f"Progress: {self.progress}/{self.max_progress} ({(self.progress/self.max_progress)*100:.1f}%)")
            
            if self.save_to_file(page_name, response.text):
                return page_name
            return None
        except Exception as e:
            print(f"Error crawling {page_name}: {e}")
            return None

    def run_batch(self, page_list, max_workers=10):
        self.max_progress = len(page_list)
        self.progress = 0
        
        print(f"Starting batch crawl for {self.max_progress} pages...")
        
        # 병렬 작업 수행
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.fetch_and_save, page_list))
        
        # 결과 분석 (성공/실패 분리)
        success_pages = [r for r in results if r is not None]
        failed_pages = [page_list[i] for i, r in enumerate(results) if r is None]
        
        success_count = len(success_pages)
        failed_count = len(failed_pages)
        
        print(f"\nBatch complete.")
        print(f" - Total: {self.max_progress}")
        print(f" - Success: {success_count}")
        print(f" - Failed: {failed_count}")

        # 실패한 리스트가 있다면 파일로 저장
        if failed_pages:
            failed_file = "failed_pages.txt"
            try:
                with open(failed_file, "w", encoding="utf-8") as f:
                    for page in failed_pages:
                        f.write(f"{page}\n")
                print(f"Failed pages list saved to: {os.path.abspath(failed_file)}")
            except Exception as e:
                print(f"Error saving failed pages list: {e}")
                
        return results
    
    def get_clean_wiki_titles(self, mmqa_file_path, texts_filepath, images_filepath, tables_filepath):
        doc_ids = set()
        img_ids = set()
        tab_ids = set()

        with open(mmqa_file_path, 'r', encoding='utf-8') as infile:
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

        final_links.update(map_ids_to_urls(texts_filepath, doc_ids))
        final_links.update(map_ids_to_urls(images_filepath, img_ids))
        final_links.update(map_ids_to_urls(tables_filepath, tab_ids))

        clean_titles = set()
        for url in final_links:
            if not url: continue
            decoded_url = urllib.parse.unquote(url)
            title = decoded_url.replace('https://en.wikipedia.org/wiki/', '').replace(' ', '_')
            clean_titles.add(title)
            
        print(f"총 {len(clean_titles)}개의 고유 wiki_title을 추출했습니다.")
        return list(clean_titles)


class BatchWikiImageCrawler:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
        }
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
        
        self.folder_path = folder_path
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.progress = 0
        self.max_progress = 0
        self.lock = threading.Lock()
    
    def fetch_and_save(self, img_data):
        filename, img_url = img_data
        file_path = os.path.join(self.folder_path, filename)
        
        if os.path.exists(file_path):
            print(f"Already Exists (Skip): {filename}")
            with self.lock:
                self.progress += 1
                if self.progress % 50 == 0:
                    print(f"Progress: {self.progress}/{self.max_progress} ({(self.progress/self.max_progress)*100:.1f}%)")
            return True
        
        try:
            time.sleep(random.randint(6,10))
        
            response = self.session.get(img_url, timeout=20, stream=True)
            if response.status_code == 429:
                print("Rate limit hit. Sleeping for 60 seconds...")
                time.sleep(60)
                return False
            
            response.raise_for_status()
            
            with self.lock:
                self.progress += 1
                print(f"Progress: {self.progress}/{self.max_progress} ({(self.progress/self.max_progress)*100:.1f}%)")
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
            return False
    
    def run_batch(self, img_data_list, max_workers=1):
        img_data_list = list(img_data_list)
        self.max_progress = len(img_data_list)
        self.progress = 0
        
        print(f"Starting batch crawl for {self.max_progress} images...")
        
        # 병렬 작업 수행
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.fetch_and_save, img_data_list))
        
        # 결과 분석 (성공/실패 분리)
        success_imgs = [r for r in results if r is not None]
        failed_imgs = [img_data_list[i] for i, r in enumerate(results) if not r]
        
        success_count = len(success_imgs)
        failed_count = len(failed_imgs)
        
        print(f"\nBatch complete.")
        print(f" - Total: {self.max_progress}")
        print(f" - Success: {success_count}")
        print(f" - Failed: {failed_count}")

        # 실패한 리스트 파일로 저장
        if failed_imgs:
            failed_file = "failed_imgs.txt"
            try:
                with open(failed_file, "w", encoding="utf-8") as f:
                    for page in failed_imgs:
                        f.write(f"{page}\n")
                print(f"Failed image list saved to: {os.path.abspath(failed_file)}")
            except Exception as e:
                print(f"Error saving failed image list: {e}")
        
        return results
    
    def get_clean_filename(self, url):
        raw_filename = url.split('/')[-1]
        decoded_name = urllib.parse.unquote(raw_filename)
        invalid_chars = '<>:"/\|?*'
        for char in invalid_chars:
            decoded_name = decoded_name.replace(char, '')        
        return decoded_name

    def extract_imglink(self, data, target_keyword="https://upload.wikimedia.org") -> set:
        links = set()
        
        if isinstance(data, (dict, list)):
            items = data.values() if isinstance(data, dict) else data
            for item in items:
                # [수정] target_keyword를 재귀 호출 시에도 전달
                links.update(self.extract_imglink(item, target_keyword))
        elif isinstance(data, str):
            # [수정] 괄호를 전체 URL에 쳐서 전체 주소를 가져오게 함
            pattern = rf"({re.escape(target_keyword)}[^\s\]]+\.(?:png|jpg|jpeg|svg))"
            matches = re.findall(pattern, data, re.IGNORECASE)
            for url in matches:
                links.add(url) # 이제 URL 전체가 저장됨
        
        return links

    def process_json_file(self, file_path):
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                image_links = self.extract_imglink(data)
                return image_links
        except:
            return None

    def get_clean_imglinks(self, filepath_list):
        links = set()
        result_links_pair = set()
        process_count = 0
        
        for path in filepath_list:
            found_urls = self.process_json_file(path)
            if found_urls is not None:
                links.update(found_urls)
                process_count += 1
        
        print(f"Processed {process_count} paths among {len(filepath_list)} paths")
        
        for url in links:
            clean_name = self.get_clean_filename(url)
            result_links_pair.add((clean_name, url))
        
        return result_links_pair
