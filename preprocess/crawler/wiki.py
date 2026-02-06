import os
import re
import json
import time
import random
import typing as tp

from concurrent.futures import ThreadPoolExecutor

import requests
from requests import Session
from requests.models import Response

from tqdm import tqdm

from common import (
    get_clean_savepath, get_clean_savepath_from_url,
    save_html_content_to_file, save_image_to_file
)

class WikiBatchCrawler:
    def __init__(
        self,
        doc_html_save_folderpath: str,
        doc_title_list: tp.List[str],
        email: str = "abc@example.com"
    ) -> None:
        assert os.path.exists(doc_html_save_folderpath)
        
        self.wiki_rest_api_url: str = f"https://en.wikipedia.org/api/rest_v1/page/html/"
        self.html_crawl_header: tp.Dict[str, str] = {
            "User-Agent": f"LILaCBulkScraper/1.0 ({email})"
        }
        
        self.doc_html_save_folderpath: str = doc_html_save_folderpath
        self.doc_title_list: tp.List[str] = sorted(doc_title_list) # Order preservation
        
        self.session: Session = requests.Session()
        self.session.headers.update(self.html_crawl_header)
        self.progress_bar = tqdm(total=0, desc="Crawling Wiki Pages")

    def _fetch_and_save(self, doc_title: str) -> bool:
        clean_savepath: str = get_clean_savepath(self.doc_html_save_folderpath, doc_title, 'html')
        if os.path.exists(clean_savepath):
            self.progress_bar.update(1)
            return True
        
        try:
            fetch_url = f"{self.wiki_rest_api_url}{doc_title}"
            
            response: Response = self.session.get(fetch_url, timeout=15)
            response.raise_for_status()
            
            self.progress_bar.update(1)
            
            if save_html_content_to_file(clean_savepath, response.text):
                return True
            return False
        except Exception as e:
            tqdm.write(f"Error crawling {doc_title}: {e}")
            return False

    def run_batch(self, failed_doc_title_list_filepath: str, max_workers=10) -> bool:
        assert self.doc_title_list == sorted(self.doc_title_list)
        
        self.progress_bar.total = len(self.doc_title_list)
        
        crawl_result_list: tp.List[bool] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            crawl_result_list = list(executor.map(self._fetch_and_save, self.doc_title_list))
        
        self.progress_bar.close()
        
        failed_doc_title_list: tp.List[str] = [self.doc_title_list[i] for i, r in enumerate(crawl_result_list) if not r]
        failed_count: int = len(failed_doc_title_list)
        success_count: int = self.progress_bar.total - failed_count
        
        print(f"\nBatch complete.")
        print(f" - Total: {self.progress_bar.total}")
        print(f" - Success: {success_count}")
        print(f" - Failed: {failed_count}")

        if failed_doc_title_list:
            try:
                with open(failed_doc_title_list_filepath, "w", encoding="utf-8") as f:
                    for page in failed_doc_title_list:
                        f.write(f"{page}\n")
                print(f"Failed doc title list saved to: {os.path.abspath(failed_doc_title_list_filepath)}")
            except Exception as e:
                print(f"Error saving failed pages list: {e}")
        
        return True


class BatchWikiImageCrawler:
    def __init__(
        self,
        image_save_folderpath: str,
        email: str = "abc@example.com"
    ) -> None:
        assert os.path.exists(image_save_folderpath)
        
        self.image_save_folderpath: str = image_save_folderpath
        self.image_crawl_header: tp.Dict[str, str] = {
            "User-Agent": f"LilacCrawler/1.0 (Contact: {email}; Research Purpose)",
            "Accept-Encoding": "gzip, deflate"
        }
        self.image_data_url_list: tp.List[str] = []
        
        self.session: Session = requests.Session()
        self.session.headers.update(self.image_crawl_header)
        
        self.progress_bar: tqdm = tqdm(total=0, desc="Crawling Wiki Images")
    
    def _fetch_and_save(self, image_url: str) -> bool:
        filename = image_url.split('/')[-1]
        clean_savepath = get_clean_savepath_from_url(self.image_save_folderpath, image_url)
        
        if os.path.exists(clean_savepath):
            self.progress_bar.update(1)
            return True

        target_urls = [image_url, f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"]
        for url in target_urls:
            try:
                time.sleep(random.uniform(0.1, 0.3))
                response = self.session.get(url, timeout=15, stream=True)
                
                if save_image_to_file(clean_savepath, response.content):
                    self.progress_bar.update(1)
                    return True
                elif response.status_code == 404:
                    continue 
            except Exception as e:
                tqdm.write(f"Error trying {url}: {e}")
                continue

        tqdm.write(f"Failed to find image on EN or Commons: {filename}")
        self.progress_bar.update(1)
        return False
    
    def run_batch(self, failed_image_list_filepath, max_workers: int = 1) -> None:
        self.progress_bar.total = len(self.image_data_url_list)
        
        crawl_result_list: tp.List[bool] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            crawl_result_list = list(executor.map(self._fetch_and_save, self.image_data_url_list))
        
        failed_image_list: tp.List[str] = [self.image_data_url_list[i] for i, r in enumerate(crawl_result_list) if not r]
        failed_count: int = len(failed_image_list)
        success_count: int = self.progress_bar.total - failed_count
        
        print(f"\nCrawl complete.")
        print(f" - Total: {self.progress_bar.total}")
        print(f" - Success: {success_count}")
        print(f" - Failed: {failed_count}\n")

        if failed_image_list:
            try:
                with open(failed_image_list_filepath, "w", encoding="utf-8") as f:
                    for image_url in failed_image_list:
                        f.write(f"{image_url}\n")
                print(f"Failed image list saved to: {os.path.abspath(failed_image_list_filepath)}")
            except Exception as e:
                print(f"Error saving failed image list: {e}")

    def set_clean_imglinks_from_folder(self, parsed_json_folderpath: str) -> bool:
        assert os.path.exists(parsed_json_folderpath)
        
        wiki_link_pattern = re.compile(r"([^\]\s\"\[]+\.(?:png|jpg|jpeg|svg|gif|webp))", re.IGNORECASE)
        image_name_set: tp.Set[str] = set()
        json_filepath_list = [f for f in os.listdir(parsed_json_folderpath) if f.endswith('.json')]
        for json_filepath in json_filepath_list:
            clean_filename = os.path.join(parsed_json_folderpath, json_filepath)
            with open(clean_filename, 'r', encoding='utf-8') as json_file:
                image_name_set.update(wiki_link_pattern.findall(json_file.read()))

        for name in list(image_name_set):
            url = f"https://en.wikipedia.org/wiki/Special:FilePath/{name.strip()}"
            self.image_data_url_list.append(url)

        self.image_data_url_list.sort()

        return True
