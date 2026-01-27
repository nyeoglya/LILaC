import re
import json
import copy
import typing as tp

import urllib.parse
from bs4 import BeautifulSoup, Tag

from base import *

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
            
            soup = BeautifulSoup(html_text, "html.parser")
            self.convert_images_to_text(soup)
            body = soup.find('body') or soup
            all_elements = body.find_all('section', recursive=False)
            
            if not all_elements:
                content_wrapper = soup.find(id="mw-content-text") or soup
                all_elements = content_wrapper.find_all('section', recursive=False)
            if not all_elements:
                return False
            
            self.source = []
            for tag in all_elements:
                classes = tag.get('class', [])
                if "shortdescription" in classes or tag.name == 'style':
                    continue
                self.source.append(tag)

            return len(self.source) > 0

        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def flatten_elements(self, elements):
        if not elements:
            return []

        flattened = []
        for el in elements:
            if el.name in {'p', 'table', 'figure', 'ul', 'ol', 'h2', 'h3', 'h4', 'h5', 'h6'}:
                flattened.append(el)
            
            elif el.name in {'div', 'section', 'center'}:
                children = [c for c in el.children if c.name]
                flattened.extend(self.flatten_elements(children))
                
        return flattened
    
    def parse_lines(self) -> bool:
        self.parsed = []
        current_heading_path = []
        stop_keywords = { "References", "See also", "External links", "Notes", "Further reading", "Sources" }
        ignore_table_classes = { 'ambox', 'mbox', 'cmbox', 'metadata' }

        for section in self.source:
            elements = section.find_all(recursive=False)
            clean_elements = self.flatten_elements(elements)
            
            for data in clean_elements:
                data_tag = data.name
                if data_tag in ['h2', 'h3', 'h4', 'h5', 'h6']:
                    title_text = data.get_text().strip()
                    
                    if any(stop.lower() in title_text.lower() for stop in stop_keywords):
                        break
                    
                    level = int(data_tag[1]) - 2 
                    current_heading_path = current_heading_path[:level]
                    current_heading_path.append(title_text)
                    continue
                
                new_component = None
                if data_tag == 'p':
                    new_component = self.parse_paragraph(data)
                elif data_tag == 'table':
                    if any(cls in ignore_table_classes for cls in data.get('class', [])):
                        continue
                    new_component = self.parse_table(data)
                elif data_tag == 'figure':
                    new_component = self.parse_figure(data)
                elif data_tag in ['ul', 'ol']:
                    new_component = self.parse_list(data)

                if new_component is not None:
                    new_component.heading_path = copy.deepcopy(current_heading_path)
                    self.parsed.append(new_component)
        return True

    def parse_paragraph(self, data: Tag) -> tp.Union[ParagraphComponent, None]:
        p_copy = copy.deepcopy(data)
        final_text, edge = self.convert_tag(p_copy)
        if len(final_text.strip()) < 1:
            return None
        else:
            return ParagraphComponent(paragraph=final_text, edge=edge)

    def parse_list(self, data: Tag) -> tp.Union[ParagraphComponent, None]:
        list_copy = copy.deepcopy(data)
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
                    indent = "  " * depth # 들여쓰기 적용
                    lines.append(f"{indent}* {item_text}")

                # 하위 리스트가 있다면 재귀적으로 처리
                for sub in sub_lists:
                    walk_list(sub, depth + 1)

        walk_list(list_copy)

        # 리스트 아이템들을 줄바꿈으로 합쳐서 하나의 문단으로 반환
        final_text = "\n".join(lines)
        return ParagraphComponent(final_text)

    def parse_table(self, data: Tag) -> TableComponent:
        table_tag = data
        rows = []
        edge_set = set()
        for tr in table_tag.find_all('tr'):
            row_data = []
            for cell in tr.find_all(['th', 'td']):
                row_text, edge = self.convert_tag(cell)
                row_data.append(row_text)
                edge_set.update(edge)
            if row_data:
                rows.append(row_data)
        
        return TableComponent(table=rows, edge=edge_set)

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
        
        edge_set = set()
        if caption_tag:
            caption_text, edge_set = self.convert_tag(caption_tag)
        
        return ImageComponent(url=full_url, caption=caption_text, edge=edge_set)

    def convert_images_to_text(self, data: Tag):
        for img in data.find_all('img'):
            alt = img.get('src', '').strip()
            full_url = "https:" + alt if alt.startswith("//") else alt
            if '/thumb/' in full_url:
                full_url = full_url.replace('/thumb/', '/')
                full_url = re.sub(r'/[^/]+$', '', full_url)
            replacement = f"[[{full_url}]]" if alt else "[Image]"
            img.replace_with(replacement)

    def convert_tag(self, data: Tag) -> tp.Tuple[str, set]:
        for ref in data.find_all(['sup', 'span'], class_='reference'): # 주석 제거
            ref.decompose()
        
        edge_set = self.convert_wikilink(data)
        
        result_text = data.get_text(separator=' ', strip=True)
        result_text = re.sub(r'\s+', ' ', result_text)
        return result_text, edge_set
    
    def convert_wikilink(self, data: Tag) -> set:
        result_edge = set()
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
                result_edge.add(page_title.split("?")[0])
                a.replace_with(link_text)
            else:
                a.unwrap()
        return result_edge