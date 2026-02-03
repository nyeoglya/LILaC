import os
import re
import json
import copy
import typing as tp

from concurrent.futures import ThreadPoolExecutor

import urllib.parse
from bs4 import BeautifulSoup, Tag
from bs4.element import ResultSet

from tqdm import tqdm

from utils import get_clean_savepath
from base import (
    ComponentData, BasePage, ParagraphComponent, ImageComponent, TableComponent
)

class WikiPage(BasePage):
    def __init__(self, doc_title: str, doc_filepath: str, doc_savepath: str) -> None:
        super().__init__()
        self.doc_title: str = doc_title
        self.doc_filepath: str = doc_filepath
        self.doc_savepath: str = doc_savepath
        self.original_html_source: tp.List[Tag] = []
        self.parsed_components: tp.List[ComponentData] = []

    def save_page(self) -> bool:
        assert self.parsed_components
        result_dict: tp.Dict[str, tp.Any] = {"title": self.doc_title, "comp_data": [parsed.to_json() for parsed in self.parsed_components]}
        tqdm.write(f"Saving {self.doc_title}...")
        with open(self.doc_savepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, separators=(',', ':'))
        return True

    def read_file(self) -> bool:
        try:
            with open(self.doc_filepath, "r", encoding="utf-8") as f:
                html_text = f.read()
            
            soup = BeautifulSoup(html_text, "html.parser")
            self._convert_images_to_text(soup)
            html_body_element = soup.find('body') or soup
            all_html_elements = html_body_element.find_all('section', recursive=False)
            
            if not all_html_elements:
                content_wrapper = soup.find(id="mw-content-text") or soup
                all_html_elements = content_wrapper.find_all('section', recursive=False)
            if not all_html_elements:
                return False
            
            self.original_html_source = []
            for element_tag in all_html_elements:
                classes = element_tag.get('class', [])
                if "shortdescription" in classes or element_tag.name == 'style':
                    continue
                self.original_html_source.append(element_tag)

            return len(self.original_html_source) > 0
        except Exception as e:
            tqdm.write(f"Error reading {self.doc_title}: {e}")
            return False
    
    def parse_lines(self) -> bool:
        self.parsed_components = []
        current_heading_path: tp.List[str] = []
        parse_stop_keywords: tp.Set[str] = { "References", "See also", "External links", "Notes", "Further reading", "Sources" }
        ignore_table_class_list: tp.Set[str] = { 'ambox', 'mbox', 'cmbox', 'metadata' }

        for original_section in self.original_html_source:
            elements_in_section: ResultSet = original_section.find_all(recursive=False)
            clean_element_list: tp.List = self._flatten_elements(elements_in_section)
            
            for clean_element in clean_element_list:
                element_tag_type: str = clean_element.name
                if element_tag_type in ['h2', 'h3', 'h4', 'h5', 'h6']:
                    title_text = clean_element.get_text().strip()
                    
                    if any(stop.lower() in title_text.lower() for stop in parse_stop_keywords):
                        break
                    
                    heading_level = int(element_tag_type[1]) - 2 
                    current_heading_path = current_heading_path[:heading_level]
                    current_heading_path.append(title_text)
                    continue
                
                new_component = None
                if element_tag_type == 'p':
                    new_component = self.parse_paragraph(clean_element)
                elif element_tag_type == 'table':
                    if any(cls in ignore_table_class_list for cls in clean_element.get('class', [])):
                        continue
                    new_component = self.parse_table(clean_element)
                elif element_tag_type == 'figure':
                    new_component = self.parse_figure(clean_element)
                elif element_tag_type in ['ul', 'ol']:
                    new_component = self.parse_list(clean_element)

                if new_component is not None:
                    new_component.heading_path = copy.deepcopy(current_heading_path)
                    self.parsed_components.append(new_component)
        return True

    def parse_paragraph(self, data: Tag) -> tp.Union[ParagraphComponent, None]:
        paragraph_copy = copy.deepcopy(data)
        final_text, edge_sets = self._convert_tag(paragraph_copy)
        if len(final_text.strip()) < 1:
            return None
        else:
            return ParagraphComponent(paragraph=final_text, edge=list(edge_sets))

    def parse_list(self, data: Tag) -> tp.Union[ParagraphComponent, None]:
        accumulated_texts, edge_sets = self._extract_list_recursive(data)
        pure_text = " ".join(accumulated_texts).replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n").strip()
        if not pure_text:
            return None
        return ParagraphComponent(paragraph=pure_text, edge=list(edge_sets))

    def parse_table(self, data: Tag) -> TableComponent:
        table_tag = data.find('tbody') or Tag()
        rows = []
        edge_set = set()
        for tr in table_tag.find_all('tr', recursive=False):
            row_data = []
            for cell in tr.find_all(['th', 'td'], recursive=False):
                row_text, edge = self._convert_tag(cell)
                row_data.append(row_text)
                edge_set.update(edge)
            if row_data:
                rows.append(row_data)
        
        return TableComponent(table=rows, edge=list(edge_set))

    def parse_figure(self, data: Tag) -> tp.Union[ImageComponent, None]:
        link_element = data.find('a')
        if link_element is None:
            return None
        src = link_element.get('href', '')
        if not src:
            return None

        filename = src.replace('./File:', '').replace('File:', '')
        filename = urllib.parse.unquote(filename)
        full_url = f"https://en.wikipedia.org/wiki/Special:FilePath/{filename}"

        caption_tag = data.find('figcaption')
        caption_text = ""
        
        edge_set = set()
        if caption_tag:
            caption_text, edge_set = self._convert_tag(caption_tag)
        
        return ImageComponent(src=full_url, caption=caption_text, edge=list(edge_set))

    def _flatten_elements(self, elements: tp.List) -> tp.List:
        if not elements:
            return []

        flattened = []
        for el in elements:
            if el.name in {'p', 'table', 'figure', 'ul', 'ol', 'h2', 'h3', 'h4', 'h5', 'h6'}:
                flattened.append(el)
            
            elif el.name in {'div', 'section', 'center'}:
                children = [c for c in el.children if c.name]
                flattened.extend(self._flatten_elements(children))
        
        return flattened

    def _extract_list_recursive(self, node: tp.Union[Tag, str]) -> tp.Tuple[tp.List[str], tp.Set[str]]:
        text_list: tp.List[str] = []
        links: tp.Set[str] = set()

        if isinstance(node, str):
            clean_text = " ".join(node.split())
            if clean_text:
                text_list.append(clean_text)
        elif node.name == "a":
            href = node.get("href") or ""
            if href.startswith("/wiki/") and ":" not in href:
                target_title = href.replace("/wiki/", "").replace("_", " ")
                links.add(target_title)
            link_text = " ".join(node.get_text().split())
            if link_text:
                text_list.append(link_text)
        elif node.name == "li":
            item_parts = []
            for child in node.children:
                child_text, child_links = self._extract_list_recursive(child)
                item_parts.extend(child_text)
                links.update(child_links)
            combined_text = "- " + " ".join(item_parts).strip() + "\n"
            return [combined_text], links
        elif node.name in ["sup", "style", "script"]:
            return [], set()
        else:
            for child in node.children:
                new_text, new_links = self._extract_list_recursive(child)
                text_list.extend(new_text)
                links.update(new_links)

        return text_list, links

    def _convert_images_to_text(self, data: Tag):
        for img in data.find_all('img'):
            res = img.get('resource')
            full_url = ""

            if res:
                file_name = res.replace('./File:', '').replace('File:', '')
                file_name = urllib.parse.unquote(file_name)
                full_url = f"https://en.wikipedia.org/wiki/Special:FilePath/{file_name}"
            else:
                src = img.get('src', '').strip()
                if not src: continue
                
                raw_url = "https:" + src if src.startswith("//") else src
                
                if 'upload.wikimedia.org' in raw_url:
                    parts = raw_url.split('/')
                    if '/thumb/' in raw_url:
                        file_name = urllib.parse.unquote(parts[-2])
                    else:
                        file_name = urllib.parse.unquote(parts[-1])
                    
                    full_url = f"https://en.wikipedia.org/wiki/Special:FilePath/{file_name}"
                else:
                    full_url = raw_url

            if full_url:
                img.replace_with(f"[[{full_url}]]")

    def _convert_tag(self, data: Tag) -> tp.Tuple[str, tp.Set[str]]:
        for ref in data.find_all(['sup', 'span'], class_='reference'): # 주석 제거
            ref.decompose()
        
        edge_set = self._convert_wikilink(data)
        
        result_text = data.get_text(separator=' ', strip=True)
        result_text = re.sub(r'\s+', ' ', result_text)
        return result_text, edge_set
    
    def _convert_wikilink(self, data: Tag) -> tp.Set[str]:
        result_edge: tp.Set[str] = set()
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

class WikiBatchParser:
    def __init__(
        self,
        doc_html_save_folderpath: str,
        doc_json_save_folderpath: str,
        doc_title_list: tp.List[str],
    ) -> None:
        assert os.path.exists(doc_json_save_folderpath)
        assert os.path.exists(doc_html_save_folderpath)
        
        self.doc_html_save_folderpath: str = doc_html_save_folderpath
        self.doc_json_save_folderpath: str = doc_json_save_folderpath
        self.doc_title_list: tp.List[str] = sorted(doc_title_list) # Order preservation
        
        self.progress_bar: tqdm = tqdm(total=0, desc="Parsing Wiki Pages")

    def _parse_and_save(self, doc_title: str) -> bool:
        json_clean_save_filepath: str = get_clean_savepath(self.doc_json_save_folderpath, doc_title, "json")
        html_doc_clean_filepath: str = get_clean_savepath(self.doc_html_save_folderpath, doc_title, "html")
        if os.path.exists(f"{json_clean_save_filepath}"):
            self.progress_bar.update(1)
            return True

        try:
            wiki_page: WikiPage = WikiPage(doc_title, html_doc_clean_filepath, json_clean_save_filepath)
            
            read_file_result = wiki_page.read_file()
            parse_lines_result = wiki_page.parse_lines()
            save_page_result = wiki_page.save_page()
            
            self.progress_bar.update(1)
            
            return read_file_result and parse_lines_result and save_page_result
        except Exception as e:
            tqdm.write(f"Error parsing {doc_title}: {e}")
            return False

    def run_batch(self, failed_doc_title_list_filepath: str, max_workers: int = 10) -> bool:
        assert self.doc_title_list == sorted(self.doc_title_list)
        
        self.progress_bar.total = len(self.doc_title_list)
        
        crawl_result_list: tp.List[bool] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            crawl_result_list = list(executor.map(self._parse_and_save, self.doc_title_list))
        
        self.progress_bar.close()
        
        failed_doc_title_list: tp.List[str] = [self.doc_title_list[i] for i, r in enumerate(crawl_result_list) if not r]
        failed_count: int = len(failed_doc_title_list)
        success_count: int = self.progress_bar.total - failed_count
        
        print(f"\nParse complete.")
        print(f" - Total: {self.progress_bar.total}")
        print(f" - Success: {success_count}")
        print(f" - Failed: {failed_count}\n")

        if failed_doc_title_list:
            try:
                with open(failed_doc_title_list_filepath, "w", encoding="utf-8") as f:
                    for page in failed_doc_title_list:
                        f.write(f"{page}\n")
                print(f"Failed doc title list saved to: {os.path.abspath(failed_doc_title_list_filepath)}")
            except Exception as e:
                print(f"Error saving failed pages list: {e}")
        
        return True
