import typing as tp

class LILaCGraph:
    def __init__(self, graph_path: str) -> None:
        self.graph_path = graph_path
    
    def load(self) -> bool:
        return False
    
    def set_graph(self) -> bool:
        return False
    
    def save(self) -> bool:
        return False
