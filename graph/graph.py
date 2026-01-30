import os
import typing as tp

import numpy as np

from processor import *

class LILaCGraph:
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.comp_embeddings: tp.List[dict] = [] # comp id -> component
        self.comp_embedding_map: np.array = np.array([]) # comp num -> comp embedding
        
        self.subcomp_embeddings: np.array = np.array([]) # comp id -> array slicing(=subcomponent_embedding)
        self.comp_idx_map: np.array = np.array([]) # comp id -> (start, end) tuple
        self.edge = tp.List[tp.List[int]] = [] # comp id -> list(comp id)

    # ------- Graph Traverse -------
    def one_hop(self, beam, beam_size) -> bool:
        return beam
    
    def find_entry(self) -> tp.List[str]:
        return []
    
    # ------- Manage Files -------
    def load(self) -> bool:
        if not os.path.exists(self.filepath):
            return False
        return True
    
    @staticmethod
    def make_graph(remapped_ldoc_path: str, filepath: str) -> bool:
        if not os.path.exists(remapped_ldoc_path):
            return False
        
        remapped_ldoc_list: tp.List[LILaCDocument] = []
        for filename in os.listdir(remapped_ldoc_path):
            if filename.endswith(".ldoc.remapped"):
                full_path = os.path.join(remapped_ldoc_path, filename)
                new_doc = LILaCDocument.load(full_path)
                remapped_ldoc_list.append(new_doc)
        
        new_lilac_graph = LILaCGraph(filepath)
        for ldoc in remapped_ldoc_list:
            for processed_comp in ldoc.processed_components:
                print(processed_comp.id)
        
        return True
    

if __name__ == "__main__":
    BEAM_SIZE = 3
    LDOC_FOLDER = "/dataset/process/mmqa/"
    GRAPH_FILE_PATH = "/dataset/process/test.lgraph"
    
    beam = [] # list(comp id)
    
    LILaCGraph.make_graph(LDOC_FOLDER)
    lilac_graph = LILaCGraph(LDOC_FOLDER, GRAPH_FILE_PATH)
