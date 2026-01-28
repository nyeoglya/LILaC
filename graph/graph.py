import os
import typing as tp

class LILaCGraph:
    def __init__(self, filepath: str, beam_size: int=3) -> None:
        self.filepath: str = filepath
        self.document_embedding_list = [] # list(doc id)
        self.document_map = {} # doc id -> list(component)
        
        self.vertex: tp.List[str] = [] # component id
        self.edge: tp.Dict[str, tp.List[str]] = dict() # vertex -> list(vertex)
        
        self.beam_size = beam_size
        self.beam = []
    
    # ------- Graph Traverse -------
    def one_hop(self) -> tp.List[str]:
        return self.beam
    
    def find_entry(self) -> tp.List[str]:
        return []
    
    # ------- Manage Files -------
    def load(self) -> bool:
        if not os.path.exists(self.filepath):
            self.set_graph()
            self.save()
        return False
    
    def set_graph(self) -> bool:
        return False
    
    def save(self) -> bool:
        return False
