import os
import re
import json
import copy
import time
import random
import typing as tp

import threading
from concurrent.futures import ThreadPoolExecutor

import urllib.parse
import requests
from bs4 import BeautifulSoup, Tag

class ComponentData:
    def __init__(self) -> None:
        self.heading_path = []
    
    def to_json(self) -> tp.Dict:
        return {}

class ImageComponent(ComponentData):
    def __init__(self, url: str, caption: str) -> None:
        super().__init__()
        self.url = url
        self.caption = caption
    
    def to_json(self) -> tp.Dict:
        return {'url': self.url, 'caption': self.caption, "heading_path": self.heading_path}
    
    def __str__(self) -> str:
        return f"url: {self.url}, caption: {self.caption}"

class ParagraphComponent(ComponentData):
    def __init__(self, paragraph: str) -> None:
        super().__init__()
        self.paragraph: str = paragraph
    
    def to_json(self) -> tp.Dict:
        return {"paragraph": self.paragraph, "heading_path": self.heading_path}
    
    def __str__(self) -> str:
        return f"paragraph: {self.paragraph}"

class TableComponent(ComponentData):
    def __init__(self, table) -> None:
        super().__init__()
        self.table = table
    
    def to_json(self) -> tp.Dict:
        return {"table": self.table, "heading_path": self.heading_path}
    
    def __str__(self) -> str:
        return f"table: {self.table}"

class BasePage:
    def __init__(self):
        self.base_url = ""
        self.source = []
    
    def save(self) -> bool:
        return False
    
    def run(self) -> bool:
        return False
    
    def parse_figure(self, data) -> tp.Union[ImageComponent, None]:
        return ImageComponent("url", "caption")
    
    def parse_paragraph(self, data: str) -> tp.Union[ParagraphComponent, None]:
        return ParagraphComponent("paragraph")
    
    def parse_table(self, data) -> tp.Union[TableComponent, None]:
        return TableComponent(data)

class WikiPage(BasePage):
    def __init__(self, title: str, filepath: str) -> None:
        super().__init__()
        self.title = title
        self.filepath = filepath
        self.source = []
        self.parsed: tp.List[ComponentData] = []

    def save(self) -> bool:
        result_list = [(0, {"title": self.title})]
        for i, comp_data in enumerate(self.parsed):
            result_list.append((i+1, comp_data.to_json()))
        result_dict = dict(result_list)
        with open(f'{self.filepath}.json', 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
        return True

    def read_file(self) -> bool:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                html_text = f.read()
            
            soup = BeautifulSoup(html_text, 'html.parser')
            
            all_elements = soup.select('section')
            
            if not all_elements:
                return False
            
            self.source = []
            seen_tags = set()
            
            for tag in all_elements:
                if tag in seen_tags:
                    continue
                
                classes = tag.get('class', [])
                if "shortdescription" in classes or tag.name == 'style':
                    continue
                    
                self.source.append(tag)
                seen_tags.add(tag)

            return len(self.source) > 0

        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def parse_lines(self) -> bool:
        self.parsed = []
        current_heading_path = []
        stop_keywords = { "References", "See also", "External links", "Notes", "Further reading", "Sources" }
        ignore_table_classes = {'ambox', 'mbox', 'cmbox', 'metadata'}

        for section in self.source:
            elements = section.find_all(recursive=False)
            header_tag = section.find(['h2', 'h3', 'h4', 'h5', 'h6'], recursive=False)
            
            if header_tag:
                title_text = header_tag.get_text().strip()
                
                if any(stop.lower() in title_text.lower() for stop in stop_keywords):
                    break
                
                level = int(header_tag.name[1]) - 2 
                current_heading_path = current_heading_path[:level]
                current_heading_path.append(title_text)

            for data in elements:
                if data == header_tag: continue
                
                new_component = None
                if data.name == 'p':
                    new_component = self.parse_paragraph(data)
                elif data.name == 'table':
                    if any(cls in ignore_table_classes for cls in data.get('class', [])):
                        continue
                    new_component = self.parse_table(data)
                elif data.name == 'figure':
                    new_component = self.parse_figure(data)
                elif data.name in ['ul', 'ol']:
                    new_component = self.parse_list(data)

                if new_component is not None:
                    new_component.heading_path = copy.copy(current_heading_path)
                    self.parsed.append(new_component)
        return True

    def parse_paragraph(self, data: Tag) -> tp.Union[ParagraphComponent, None]:
        p_copy = copy.copy(data)
        final_text = self.convert_tag(p_copy)
        if len(final_text.strip()) < 0:
            return None
        else:
            return ParagraphComponent(final_text)

    def parse_list(self, data: Tag) -> tp.Union[ParagraphComponent, None]:
        list_copy = copy.copy(data)
        lines = []

        def walk_list(tag: Tag, depth: int = 0):
            for li in tag.find_all('li', recursive=False):
                for ref in li.find_all(['sup', 'span'], class_='reference'):
                    ref.decompose()
                self.convert_wikilink(li)

                sub_lists = li.find_all(['ul', 'ol'], recursive=False)
                for sub in sub_lists:
                    sub.extract() # li 텍스트 추출을 위해 잠시 분리

                # 현재 li의 순수 텍스트 추출 및 정리
                item_text = li.get_text(separator=' ', strip=True)
                item_text = re.sub(r'\s+', ' ', item_text)

                if item_text:
                    indent = "  " * depth  # 들여쓰기 적용
                    lines.append(f"{indent}* {item_text}")

                # 하위 리스트가 있다면 재귀적으로 처리
                for sub in sub_lists:
                    walk_list(sub, depth + 1)

        walk_list(list_copy)

        # 리스트 아이템들을 줄바꿈으로 합쳐서 하나의 문단으로 반환
        final_text = "\n".join(lines)
        return ParagraphComponent(final_text)

    def parse_table(self, data: Tag) -> TableComponent:
        table_tag = copy.copy(data)
        rows = []
        
        for tr in table_tag.find_all('tr'):
            row_data = []
            for cell in tr.find_all(['th', 'td']):
                row_data.append(self.convert_tag(cell))
            
            if row_data:
                rows.append(row_data)
                
        return TableComponent(rows)

    def parse_figure(self, data) -> tp.Union[ImageComponent, None]:
        img_element = data.find('img')
        if img_element is None:
            return None
        src = img_element.get('src', '')
        if not src:
            return None

        full_url = "https:" + src if src.startswith("//") else src
        
        if '/thumb/' in full_url:
            full_url = full_url.replace('/thumb/', '/')
            full_url = re.sub(r'/[^/]+$', '', full_url)

        caption_tag = data.find('figcaption')
        caption_text = ""
        
        if caption_tag:
            caption_text = self.convert_tag(caption_tag)
        
        return ImageComponent(url=full_url, caption=caption_text)

    def convert_images_to_text(self, data: Tag):
        for img in data.find_all('img'):
            alt = img.get('src', '').strip()
            full_url = "https:" + alt if alt.startswith("//") else alt        
            if '/thumb/' in full_url:
                full_url = full_url.replace('/thumb/', '/')
                full_url = re.sub(r'/[^/]+$', '', full_url)
            replacement = f"[[Image:{full_url}]]" if alt else "[Image]"
            img.replace_with(replacement)

    def convert_tag(self, data):
        for ref in data.find_all(['sup', 'span'], class_='reference'): # 주석 제거
            ref.decompose()
        
        self.convert_images_to_text(data)
        self.convert_wikilink(data)
        
        result_text = data.get_text(separator=' ', strip=True)
        result_text = re.sub(r'\s+', ' ', result_text)
        return result_text
    
    def convert_wikilink(self, data):
        for a in list(data.find_all('a')):
            href = a.get('href', '')
            rel = a.get('rel', [])
            link_text = a.get_text().strip()
            
            if not link_text:
                a.decompose()
                continue

            is_wikilink = "mw:WikiLink" in rel or href.startswith('./')
            is_file = "File:" in href or "Special:" in href

            if is_wikilink and not is_file:
                page_title = urllib.parse.unquote(href.replace('./', '').replace('/wiki/', ''))
                
                if page_title != link_text:
                    new_syntax = f"[[{page_title}|{link_text}]]"
                else:
                    new_syntax = f"[[{link_text}]]"
                
                a.replace_with(new_syntax)
            else:
                a.unwrap()

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

    def run_batch(self, page_list, max_workers=5):
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
    
    def get_clean_wiki_titles(self, mmqa_file_path):
        titles = set()
        
        with open(mmqa_file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue # 빈 줄 건너뛰기
                
                try:
                    # 한 줄씩 읽어서 딕셔너리로 변환
                    entry = json.loads(line)
                    
                    # 만약 파일 전체가 [{}, {}] 구조라면 entry가 리스트일 수 있음
                    if isinstance(entry, list):
                        data_list = entry
                    else:
                        data_list = [entry]

                    for item in data_list:
                        # 데이터가 문자열이면 무시 (에러 방지)
                        if not isinstance(item, dict):
                            continue
                            
                        metadata = item.get('metadata', {})
                        
                        # wiki_entities_in_answers에서 추출
                        for entity in metadata.get('wiki_entities_in_answers', []):
                            if isinstance(entity, dict):
                                title = entity.get('wiki_title')
                                if title: titles.add(title.strip())
                        
                        # wiki_entities_in_question에서 추출
                        for entity in metadata.get('wiki_entities_in_question', []):
                            if isinstance(entity, dict):
                                title = entity.get('wiki_title')
                                if title: titles.add(title.strip())

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON on line {line_number}")
                    continue

        # 공백을 언더바로 치환
        final_list = [t.replace(' ', '_') for t in titles if t]
        print(f"총 {len(final_list)}개의 고유 wiki_title을 추출했습니다.")
        return final_list

class BatchWikiImageCrawler:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Referer": "https://en.wikipedia.org/"
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
            time.sleep(random.uniform(1.0, 3.0)) 
        
            response = self.session.get(img_url, timeout=15, stream=True)
            if response.status_code == 429:
                print("Rate limit hit. Sleeping for 5 seconds...")
                time.sleep(5.0)
                return False
            
            response.raise_for_status()
            
            with self.lock:
                self.progress += 1
                if self.progress % 50 == 0:
                    print(f"Progress: {self.progress}/{self.max_progress} ({(self.progress/self.max_progress)*100:.1f}%)")
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
            return False
    
    def run_batch(self, img_data_list, max_workers=2):
        img_data_list = list(img_data_list)
        self.max_progress = len(img_data_list)
        self.progress = 0
        
        print(f"Starting batch crawl for {self.max_progress} images...")
        
        # 병렬 작업 수행
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.fetch_and_save, img_data_list))
        
        # 결과 분석 (성공/실패 분리)
        success_pages = [r for r in results if r is not None]
        failed_pages = [img_data_list[i] for i, r in enumerate(results) if r is None]
        
        success_count = len(success_pages)
        failed_count = len(failed_pages)
        
        print(f"\nBatch complete.")
        print(f" - Total: {self.max_progress}")
        print(f" - Success: {success_count}")
        print(f" - Failed: {failed_count}")

        # 실패한 리스트가 있다면 파일로 저장
        if failed_pages:
            failed_file = "failed_imgs.txt"
            try:
                with open(failed_file, "w", encoding="utf-8") as f:
                    for page in failed_pages:
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

    def extract_imglink(self, data, target_keyword="https://upload.wikimedia.org") -> tp.Set[str]:
        links = set()
        
        if isinstance(data, dict):
            for value in data.values():
                links.update(self.extract_imglink(value, target_keyword))
        elif isinstance(data, list):
            for item in data:
                links.update(self.extract_imglink(item, target_keyword))
        elif isinstance(data, str):
            # [[Image:URL]] 형태 (문자열 내부에 포함됨)
            if "[[Image:" in data:
                found = re.findall(r'\[\[Image:(.*?)\]\]', data)
                for link in found:
                    if target_keyword in link:
                        links.add(link.strip())
            
            # URL만 단독으로 있는 경우 (또는 단순 포함)
            elif target_keyword in data:
                found = re.findall(rf'({target_keyword}[^\s\]|]+)', data)
                links.update(found)
                
        return links

    def process_json_file(self, file_path):
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                target_data = list(data.values())
                image_links = self.extract_imglink(target_data)
                return image_links
        except:
            return None

    def get_clean_imglinks(self, filepath_list):
        links = set()
        result_links_pair = set()
        process = 0
        for data in filepath_list:
            processed_file = self.process_json_file(data)
            if processed_file is not None:
                links.update(processed_file)
                process += 1
        
        print(f"Process {process} paths among {len(filepath_list)} paths")
        
        for link in links:
            result_links_pair.add((self.get_clean_filename(link), link))
        
        return result_links_pair
