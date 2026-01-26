import os
import re
import copy
import typing as tp

import threading
from concurrent.futures import ThreadPoolExecutor

import urllib.parse
import requests
from bs4 import BeautifulSoup, Tag

class ComponentData:
    def __init__(self) -> None:
        self.heading_path = []

class ImageComponent(ComponentData):
    def __init__(self, url: str, caption: str) -> None:
        super().__init__()
        self.url = url
        self.caption = caption
    
    def __str__(self) -> str:
        return f"url: {self.url}, caption: {self.caption}"

class ParagraphComponent(ComponentData):
    def __init__(self, paragraph: str) -> None:
        super().__init__()
        self.paragraph: str = paragraph
    
    def __str__(self) -> str:
        return f"paragraph: {self.paragraph}"

class TableComponent(ComponentData):
    def __init__(self, table) -> None:
        super().__init__()
        self.table = table
    
    def __str__(self) -> str:
        return f"table: {self.table}"

class LilacCrawlerBase:
    def __init__(self):
        self.base_url = ""
        self.source = []
    
    def run(self) -> bool:
        return False
    
    def parse_figure(self, data) -> tp.Union[ImageComponent, None]:
        return ImageComponent("url", "caption")
    
    def parse_paragraph(self, data: str) -> tp.Union[ParagraphComponent, None]:
        return ParagraphComponent("paragraph")
    
    def parse_table(self, data) -> tp.Union[TableComponent, None]:
        return TableComponent(data)

class WikiPage(LilacCrawlerBase):
    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.filepath = filepath
        self.source = []
        self.parsed: tp.List[ComponentData] = []

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
        img_tag = copy.copy(data)
        
        src = img_tag.find('img').get('src', '')
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
        for a in data.find_all('a'):
            href = a.get('href', '')
            link_text = a.get_text(strip=True)
            if href.startswith('/wiki/') and not href.startswith('/wiki/File:'):
                page_title = urllib.parse.unquote(href.replace('/wiki/', ''))
                new_syntax = f"[[{page_title}|{link_text}]]" if page_title != link_text else f"[[{link_text}]]"
                a.replace_with(new_syntax)
            else:
                a.unwrap()

class WikiBatchParser:
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

    def save_to_file(self, page_name, html_content):
        safe_filename = "".join([c for c in page_name if c.isalnum() or c in (' ', '_', '-')]).rstrip()
        file_path = os.path.join(self.folder_path, f"{safe_filename}.html")
        
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
                print(f"⚠️ Failed pages list saved to: {os.path.abspath(failed_file)}")
            except Exception as e:
                print(f"Error saving failed pages list: {e}")
                
        return results
