import typing as tp

class ComponentData:
    def __init__(self, edge: set=set()) -> None:
        self.heading_path = []
        self.edge = list(edge)
    
    def to_json(self) -> tp.Dict:
        return {"heading_path": self.heading_path, "edge": self.edge}

class ImageComponent(ComponentData):
    def __init__(self, src: str, caption: str, edge: set=set()) -> None:
        super().__init__(edge)
        self.src = src
        self.caption = caption
    
    def to_json(self) -> tp.Dict:
        return {"type": "image", 'src': self.src, 'caption': self.caption, "heading_path": self.heading_path, "edge": self.edge}
    
    def __str__(self) -> str:
        return f"src: {self.src}, caption: {self.caption}"

class ParagraphComponent(ComponentData):
    def __init__(self, paragraph: str, edge: set=set()) -> None:
        super().__init__(edge)
        self.paragraph: str = paragraph
    
    def to_json(self) -> tp.Dict:
        return {"type": "paragraph", "paragraph": self.paragraph, "heading_path": self.heading_path, "edge": self.edge}
    
    def __str__(self) -> str:
        return f"paragraph: {self.paragraph}"

class TableComponent(ComponentData):
    def __init__(self, table: list, edge: set=set()) -> None:
        super().__init__(edge)
        self.table = table
    
    def to_json(self) -> tp.Dict:
        return {"type": "table", "table": self.table, "heading_path": self.heading_path, "edge": self.edge}
    
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
        return ImageComponent("src", "caption")
    
    def parse_paragraph(self, data: str) -> tp.Union[ParagraphComponent, None]:
        return ParagraphComponent("paragraph")
    
    def parse_table(self, data) -> tp.Union[TableComponent, None]:
        return TableComponent(data)
