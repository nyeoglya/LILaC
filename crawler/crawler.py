import re
import copy
import typing as tp
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

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
        self.headers = {
            'User-Agent': 'LILaCScraper/1.0 (abc@example.com)'
        }
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
    def __init__(self, page_name: str) -> None:
        super().__init__()
        self.page_name = page_name
        self.source = []
        self.parsed: tp.List[ComponentData] = []
        self.base_url = "https://en.wikipedia.org/w/api.php"

    def run(self) -> bool:
        self.crawl_restapi()
        self.parse_lines()
        return True

    def crawl_old(self) -> bool:
        params = {
            "action": "parse",
            "page": self.page_name,
            "prop": "text",
            "format": "json",
            "redirects": 1,
            "disableeditsection": 1
        }
        
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                print(f"API Error: {data['error'].get('info', 'Unknown error')}")
                return False

            raw_html = data.get('parse', {}).get('text', {}).get('*')
            
            if raw_html:
                soup = BeautifulSoup(raw_html, 'html.parser')
                container = soup.find("div", class_="mw-parser-output")
                
                if not container:
                    return False

                self.source = [tag for tag in container.find_all(recursive=False) if tag.name and "shortdescription" not in tag.get('class', []) and tag.name != 'style']

                return True
            return False

        except Exception as e:
            print(f"Error: {e}")
            return False

    def crawl_restapi(self) -> bool:
        url = f"https://en.wikipedia.org/api/rest_v1/page/html/{self.page_name}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            all_elements = soup.select('section > p, section > table, section > div, section > ul, section > figure')
            
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
        current_heading_index = 0
        heading_map = {
            'mw-heading2': 0,
            'mw-heading3': 1,
            'mw-heading4': 2,
            'mw-heading5': 3,
            'mw-heading6': 4,
        }
        ignore_table_classes = {'ambox', 'mbox', 'cmbox', 'infobox-servicedisruption'}
        stop_keywords = { "References", "See also", "External links", "Notes", "Further reading", "Sources" }
        for data in self.source:
            new_component: tp.Union[ComponentData, None] = None
            
            if data.name == 'div':
                class_list = data.get('class', [])
                if 'mw-heading' in class_list:
                    found_class = next((c for c in class_list if c in heading_map), None)
                    if found_class:
                        current_heading_index = heading_map[found_class]
                        current_heading_path = current_heading_path[:current_heading_index]
                        title_text = data.text.strip()
                        if title_text in stop_keywords:
                            break
                        else:
                            current_heading_path.append(title_text)
            elif data.name == "dl":
                current_heading_path = current_heading_path[:current_heading_index+1]
                current_heading_path.append(data.text)
            elif data.name == 'p':
                new_component = self.parse_paragraph(data)
            elif data.name == 'table':
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

class WikiBatchHTMLParser:
    def __init__(self, lang="en"):
        self.base_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/"
        self.headers = {
            "User-Agent": "WikiBulkParser/1.0 (your-email@example.com)"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def parse_page(self, page_name):
        url = self.base_url + page_name
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            container = soup.find("body")
            
            if not container:
                return None

            return {
                "page": page_name,
                "html_content": str(container)
            }

        except Exception as e:
            print(f"Error parsing {page_name}: {e}")
            return None

    def run_batch(self, page_list, max_workers=3):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.parse_page, page_list))
        
        return [r for r in results if r is not None]
