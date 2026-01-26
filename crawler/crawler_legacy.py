import requests
import wikitextparser as wtp
import typing as tp
import pandas as pd
import re

class NotParsedError(Exception):
    def __init__(self, message="You should execute the method run() to parse the data."):
        super().__init__(message)

class LilacCrawlerBase:
    def __init__(self):
        self.headers = {
            'User-Agent': 'LILaCScraper/1.0 (abc@example.com)'
        }
        self.base_url = ""
    
    def run(self) -> bool:
        return False
    
    def parse_imagelink(self) -> tp.List[str]:
        return []
    
    def parse_paragraph(self) -> tp.List[str]:
        return []
    
    def parse_table(self) -> tp.List[pd.DataFrame]:
        return []
    
class WikiPageCrawler(LilacCrawlerBase):
    def __init__(self, page_name: str) -> None:
        super().__init__()
        self.page_name = page_name
        self.parsed = None
        self.base_url = "https://en.wikipedia.org/w/api.php"

    def run(self) -> bool:
        params = {
            "action": "parse",
            "page": self.page_name,
            "prop": "wikitext",
            "format": "json"
        }
        
        response = requests.get(self.base_url, params=params, headers=self.headers)
        
        if response.status_code != 200:
            print(f"Crawler Error: Status code {response.status_code}. response: {response.text}")
            return False

        try:
            data = response.json()
            if 'error' in data:
                print(f"API Error: {data['error']['info']}")
                return False
            self.parsed = wtp.parse(data['parse']['wikitext']['*'])
            return True
        except requests.exceptions.JSONDecodeError:
            print(f"Crawler Error: Cannot parse JSON. response: {response.text}")
            return False

    def parse_imagelink(self) -> tp.List[str]:
        if self.parsed is None:
            raise NotParsedError
        
        # 위키 텍스트 내의 [[File:Example.jpg]] 또는 [[Image:Example.jpg]] 추출
        images = [tag.title for tag in self.parsed.wikilinks if tag.title.startswith(('File:', 'Image:'))]
        return images
    
    def parse_paragraph(self) -> tp.List[str]:
        if self.parsed is None:
            raise NotParsedError()

        exclude_sections = ['References', 'External links', 'See also', 'Further reading', 'Notes']
        clean_paragraphs = []

        for section in self.parsed.sections:
            title = section.title.strip() if section.title else ""
            if any(ex.lower() in title.lower() for ex in exclude_sections):
                break
            
            sect_copy = wtp.parse(section.string)
            for table in sect_copy.tables:
                table.string = ""

            raw_sect_text = sect_copy.string
            protected = re.sub(r'\[\[([^|\]]+)\]\]', r'@L_START@\1@L_END@', raw_sect_text)
            protected = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'@L_START@\1|\2@L_END@', protected)
            content = wtp.parse(protected).plain_text()
            restored = content.replace('@L_START@', '[[').replace('@L_END@', ']]')

            for chunk in restored.split('\n\n'):
                paragraph = chunk.replace('\n', ' ').strip()
                
                paragraph = re.sub(r'<ref.*?>.*?</ref>', '', paragraph, flags=re.DOTALL)
                paragraph = re.sub(r'<ref.*?/>', '', paragraph)
                paragraph = re.sub(r'={3,}.*?={3,}', '', paragraph).strip()
                paragraph = re.sub(r'={2,}.*?={2,}', '', paragraph).strip()
                paragraph = re.sub(r'\[\[(?:Image|File):.*?\]\]', '', paragraph).strip()
                
                if not paragraph:
                    continue
                elif re.match(r'^(\{\||\|\s*\}|\|\s*-|\|\s*\+)\s*$', paragraph):
                    continue
                elif paragraph.startswith('==') and paragraph.endswith('=='):
                    continue
                
                if paragraph.startswith((';', '*')):
                    if '{|' in paragraph or '|}' in paragraph:
                        continue
                    
                    category = ""
                    if ';' in paragraph:
                        match = re.search(r';(.*?)\*', paragraph)
                        category = match.group(1).strip() if match else ""
                    
                    items = [it.strip() for it in paragraph.split('*') if it.strip() and it.strip() != category]
                    
                    if items:
                        if len(items) > 1:
                            joined_items = ", ".join(items[:-1]) + " and " + items[-1]
                        else:
                            joined_items = items[0]
                            
                        category_str = f" {category}" if category else ""
                        verbalized = f"{title}'s{category_str} includes {joined_items}."
                        
                        clean_paragraphs.append(verbalized)
                    
                    continue

                if len(paragraph) > 10 or '<math' in paragraph:
                    if not (paragraph.startswith('|') and '=' in paragraph and len(paragraph) < 100):
                        clean_paragraphs.append(paragraph)

        return clean_paragraphs
    
    def parse_table(self) -> tp.List[pd.DataFrame]:
        if self.parsed is None:
            raise NotParsedError
        
        tables = []
        for table in self.parsed.tables:
            table_data = table.data()
            if table_data:
                tables.append(table_data)
        
        return tables

    def parse_infobox(self) -> tp.List[pd.DataFrame]:
        if self.parsed is None:
            return []
        
        infobox_data = []
        
        # 1. 모든 템플릿 중 'Infobox' 탐색
        for template in self.parsed.templates:
            if template.name.strip().lower().startswith('infobox'):
                # 2. 각 인포박스마다 내부 데이터를 담을 리스트
                current_infobox = []
                
                for arg in template.arguments:
                    key = arg.name.strip()
                    # 3. 위키 마크업을 제거하고 순수 텍스트만 추출
                    val_parsed = wtp.parse(arg.value)
                    value = val_parsed.plain_text().strip()
                    
                    # 셀 내부 리스트(*)가 있다면 리스트 형태로 변환 (원하신다면)
                    if '*' in arg.value:
                        value = [it.strip() for it in value.split('*') if it.strip()]
                    
                    current_infobox.append([key, value])
                
                # 여러 개의 인포박스가 있을 수 있으므로 리스트에 추가
                infobox_data.append(current_infobox)
                    
        return infobox_data

class BatchCrawling:
    def __init__(self) -> None:
        pass
    
    def run(self) -> None:
        pass
