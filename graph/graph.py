import os
import typing as tp

class LightWeightComponent:
    def __init__(self, embedding, subcomponent_embedding):
        self.embedding = embedding
        self.subcomponent_embedding = subcomponent_embedding

class LILaCGraph:
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.document_map = {} # doc embedding -> list(comp id)
        self.component_map = {} # comp id -> processed component
        
        self.vertex: tp.Dict[str, LightWeightComponent] = [] # comp id -> comp vector data
        self.edge: tp.Dict[str, tp.List[str]] = dict() # comp id -> list(comp id)

    # ------- Graph Traverse -------
    def one_hop(self, beam, beam_size) -> bool:
        return beam
    
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

if __name__ == "__main__":
    BEAM_SIZE = 3
    beam = [] # list(comp id)
    
    lilac_graph = LILaCGraph("???")
