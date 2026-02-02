import typing as tp

from dataclasses import dataclass, field, asdict
import typing as tp
from bs4 import Tag

@dataclass
class ComponentData:
    heading_path: tp.List[str] = field(default_factory=list)
    edge: tp.List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.edge, set):
            self.edge = list(self.edge)

    def to_json(self) -> tp.Dict:
        return asdict(self)

dataclass
class ParagraphComponent(ComponentData):
    paragraph: str = ""

    def to_json(self) -> tp.Dict:
        return {"type": "paragraph", **asdict(self)}

@dataclass
class ImageComponent(ComponentData):
    src: str = ""
    caption: str = ""

    def to_json(self) -> tp.Dict:
        return {"type": "image", **asdict(self)}

@dataclass
class TableComponent(ComponentData):
    table: tp.List = field(default_factory=list)

    def to_json(self) -> tp.Dict:
        return {"type": "table", **asdict(self)}

class BasePage:
    def __init__(self):
        self.base_url = ""
        self.source = []
    
    def save(self) -> bool:
        return False
    
    def run(self) -> bool:
        return False
    
    def parse_figure(self, data: Tag) -> tp.Union[ImageComponent, None]:
        return None
    
    def parse_paragraph(self, data: Tag) -> tp.Union[ParagraphComponent, None]:
        return None
    
    def parse_table(self, data: Tag) -> tp.Union[TableComponent, None]:
        return None
