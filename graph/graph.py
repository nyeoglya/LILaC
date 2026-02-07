import os
import pickle
import typing as tp

import numpy as np
from tqdm import tqdm

from embed import ProcessedComponent, LILaCDocument

class LILaCGraph:
    def __init__(self) -> None:
        self.component_id_map: tp.List[tp.Dict] = [] # comp id -> component
        self.component_doc_title_map: tp.List[str] = [] # comp id -> doc title, debugging purpose
        self.component_embedding_map: np.ndarray = np.array([]) # comp id -> comp embedding
        self.component_uuid_map: tp.List[tp.List[str]] = [] # comp id -> uuid (for retrieval score)
        
        self.subcomponent_embeddings_dump: np.ndarray = np.array([]) # comp id -> array slicing(=subcomponent_embedding)
        self.subcomponent_embedding_range_map: np.ndarray = np.array([]) # comp id -> (start, end) tuple
        self.component_edge_map: tp.List[tp.Set[int]] = [] # comp id -> set(comp id)

    @staticmethod
    def load_graph(graph_filepath: str) -> "LILaCGraph":
        if not os.path.exists(graph_filepath):
            raise FileNotFoundError
        
        with open(graph_filepath, "rb") as f:
            lilac_graph: LILaCGraph = pickle.load(f)
            # edge_count = sum([len(component_edge_list) for component_edge_list in lilac_graph.component_edge_map])
            # print(f"Complete loading graph. Total {len(lilac_graph.component_id_map)} components, {edge_count} edges")
            return lilac_graph

    @staticmethod
    def make_graph(lilac_doc_folderpath: str, graph_filepath: str) -> bool:
        assert os.path.exists(lilac_doc_folderpath)

        lilac_doc_filename_list: tp.List[str] = [filename for filename in os.listdir(lilac_doc_folderpath)]
        lilac_doc_list_with_none: tp.List[tp.Optional[LILaCDocument]] = [LILaCDocument.load_from_path(os.path.join(lilac_doc_folderpath, f)) for f in lilac_doc_filename_list]
        lilac_doc_list: tp.List[LILaCDocument] = [lilac_doc for lilac_doc in lilac_doc_list_with_none if lilac_doc is not None]

        processed_component_list: tp.List[ProcessedComponent] = []
        for lilac_doc in lilac_doc_list:
            processed_component_list.extend(lilac_doc.processed_components)
        
        total_component_count: int = len(processed_component_list)
        embedding_dimension = processed_component_list[0].component_embedding.shape[0]
        total_subcomponent_count = sum(len(processed_component.subcomponent_embeddings) for processed_component in processed_component_list)
        
        if total_component_count == 0: return False
        component_id_map: tp.List[tp.Optional[tp.Dict]] = [None] * total_component_count
        component_doc_title_map: tp.List[tp.Optional[str]] = [None] * total_component_count
        component_uuid_map: tp.List[tp.List[str]] = [[]] * total_component_count
        component_edge_map: tp.List[tp.Set] = [set() for _ in range(total_component_count)]
        subcomponent_embedding_range_map: np.ndarray = np.zeros((total_component_count, 2), dtype=np.int64)
        
        component_embedding_map = np.empty((total_component_count, embedding_dimension), dtype=np.float32)
        subcomponent_embeddings_dump = np.empty((total_subcomponent_count, embedding_dimension), dtype=np.float32)

        subcomponent_id_cursor = 0
        current_component_id = 0
        doc_title_component_id_list_map: tp.Dict[str, tp.List[int]] = dict()
        for lilac_doc in tqdm(lilac_doc_list, desc="Component Indexing..."):
            inter_component_id_list = [current_component_id + i for i in range(len(lilac_doc.processed_components))]
            doc_title_component_id_list_map[lilac_doc.doc_title] = inter_component_id_list
            for processed_component in lilac_doc.processed_components:
                component_id_map[current_component_id] = processed_component.original_component
                component_doc_title_map[current_component_id] = lilac_doc.doc_title
                component_embedding_map[current_component_id] = processed_component.component_embedding
                component_uuid_map[current_component_id] = processed_component.component_uuid

                for other_component_id in inter_component_id_list: # inter connection
                    component_edge_map[current_component_id].add(other_component_id)
                    component_edge_map[other_component_id].add(current_component_id)

                subcomponent_embedding_list = processed_component.subcomponent_embeddings
                start_id, end_id = subcomponent_id_cursor, subcomponent_id_cursor + len(subcomponent_embedding_list)
                subcomponent_embedding_range_map[current_component_id] = (start_id, end_id)
                try:
                    subcomponent_embeddings_dump[start_id:end_id] = np.array(subcomponent_embedding_list)
                except:
                    print(subcomponent_embedding_list)
                    print(lilac_doc.doc_title, processed_component.original_component)
                subcomponent_id_cursor = end_id
                current_component_id += 1
        
        current_component_id = 0
        for lilac_doc in tqdm(lilac_doc_list, desc="Outer Component Id Mapping..."):
            for processed_component in lilac_doc.processed_components:
                if processed_component.neighbor_components: # outer component id mapping
                    for neighbor_component in processed_component.neighbor_components:
                        if neighbor_component in doc_title_component_id_list_map:
                            component_edge_map[current_component_id].update(doc_title_component_id_list_map[neighbor_component])
                current_component_id += 1

        assert None not in component_id_map
        assert None not in component_doc_title_map

        graph = LILaCGraph()
        graph.component_id_map = tp.cast(tp.List[tp.Dict], component_id_map)
        graph.component_doc_title_map = tp.cast(tp.List[str], component_doc_title_map)
        graph.component_embedding_map = component_embedding_map
        graph.subcomponent_embeddings_dump = subcomponent_embeddings_dump
        graph.subcomponent_embedding_range_map = subcomponent_embedding_range_map
        graph.component_edge_map = component_edge_map
        graph.component_uuid_map = component_uuid_map
        
        with open(graph_filepath, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL) # HIGHEST_PROTOCOL: Save large object (>4GB)

        return True

class LILaCBeam:
    def __init__(self, lilac_graph: LILaCGraph) -> None:
        self.lilac_graph: LILaCGraph = lilac_graph
        
        self.beam_size: tp.Optional[int] = None
        self.max_hop: tp.Optional[int] = None
        self.subquery_embeddings: tp.Optional[np.ndarray] = None
        
        self.beam: tp.List[int] = []

    def reset_values(self, beam_size: int, max_hop: int, subquery_embeddings: np.ndarray):
        self.beam_size = beam_size
        self.max_hop = max_hop
        self.subquery_embeddings = subquery_embeddings
        self.beam = []

    def find_entry(self, original_embedding: np.ndarray) -> bool:
        assert self.beam_size
        assert self.subquery_embeddings is not None
        
        component_scores = self.lilac_graph.component_embedding_map @ original_embedding.T
        top_2048_indices = np.argsort(-component_scores, axis=0)[:2048]
        
        component_candidate_list = []
        for component_id in top_2048_indices:
            start_pos, end_pos = self.lilac_graph.subcomponent_embedding_range_map[component_id]
            if start_pos < end_pos:
                subcomponent_scores = self.lilac_graph.subcomponent_embeddings_dump[start_pos:end_pos] @ self.subquery_embeddings.T
                component_candidate_list.append((subcomponent_scores.max(), component_id))
        
        component_candidate_list.sort(key=lambda x: x[0], reverse=True)
        self.beam = [item[1] for item in component_candidate_list[:self.beam_size]]
        return len(self.beam) > 0

    # def find_entry(self, original_embedding: np.ndarray) -> bool:
    #     assert self.beam_size
    #     assert self.subquery_embeddings is not None
        
    #     component_scores = (self.lilac_graph.component_embedding_map @ original_embedding.T).flatten()
    #     top_2048_indices = np.argsort(-component_scores)[:2048]
        
    #     component_candidate_list = []
    #     for component_id in top_2048_indices:
    #         macro_score = float(component_scores[component_id])
    #         component_candidate_list.append((macro_score, component_id))
        
    #     component_candidate_list.sort(key=lambda x: x[0], reverse=True)
    #     self.beam = [item[1] for item in component_candidate_list[:self.beam_size]]
        
    #     return len(self.beam) > 0

    def one_hop(self) -> bool:
        assert self.beam
        assert self.beam_size

        candidate_edge_list = {}
        for component_id in self.beam:
            neighbor_id_list = self.lilac_graph.component_edge_map[component_id]
            if not neighbor_id_list:
                candidate_edge_list[frozenset([component_id])] = self._calculate_maxsim_score(component_id)
                continue
            
            for neighbor_id in neighbor_id_list:
                candidate_edge = frozenset([component_id, neighbor_id])
                if candidate_edge in candidate_edge_list: continue
                
                candidate_edge_score = self._calculate_maxsim_score(component_id, neighbor_id)
                component1_single_score = self._calculate_maxsim_score(component_id)
                component2_single_score = self._calculate_maxsim_score(neighbor_id)

                if candidate_edge_score <= max(component1_single_score, component2_single_score) + 1e-5: # epsilon = 1e-5 for numerical stability
                    winner_id = component_id if component1_single_score >= component2_single_score else neighbor_id
                    candidate_edge_list[frozenset([winner_id])] = max(component1_single_score, component2_single_score)
                else:
                    candidate_edge_list[candidate_edge] = candidate_edge_score

        sorted_candidate_edge_list = sorted(candidate_edge_list.items(), key=lambda x: x[1], reverse=True)

        new_beam = []
        seen = set()
        for component_id_set, _ in sorted_candidate_edge_list:
            for cid in component_id_set:
                clean_id = int(cid) 
                if clean_id not in seen:
                    seen.add(clean_id)
                    new_beam.append(clean_id)
            if len(new_beam) >= self.beam_size:
                break
        
        old_beam, self.beam = self.beam, new_beam
        return old_beam != self.beam

    def multi_hop(self):
        assert self.max_hop
        hop_count = 0
        while hop_count < self.max_hop:
            changed = self.one_hop()
            if not changed: break
            hop_count += 1
    
    def _calculate_maxsim_score(self, component1_id: int, component2_id: tp.Optional[int] = None) -> float:
        start1, end1 = self.lilac_graph.subcomponent_embedding_range_map[component1_id]
        embedding1 = self.lilac_graph.subcomponent_embeddings_dump[start1:end1]
        
        if component2_id is not None and component1_id != component2_id:
            start2, end2 = self.lilac_graph.subcomponent_embedding_range_map[component2_id]
            embedding2 = self.lilac_graph.subcomponent_embeddings_dump[start2:end2]
            combined_subembeddings = np.concatenate((embedding1, embedding2), axis=0)
        else:
            combined_subembeddings = embedding1

        if combined_subembeddings.size == 0: return -1.0

        sim_matrix = self.subquery_embeddings @ combined_subembeddings.T 
        return float(np.sum(np.max(sim_matrix, axis=1)))

    def top_component_id_list(self, top_k: int) -> tp.List[int]:
        assert self.beam
        return self.beam[:top_k]

    def top_component_list(self, top_k: int) -> tp.List[dict]:
        top_id_list: tp.List[int] = self.top_component_id_list(top_k)
        return [self.lilac_graph.component_id_map[cid] for cid in top_id_list]
    
    def top_component_uuid_set(self, top_k: int) -> tp.Set[str]:
        top_id_list: tp.List[int] = self.top_component_id_list(top_k)
        result_set: tp.Set[str] = set()
        for component_id in top_id_list:
            result_set.update(self.lilac_graph.component_uuid_map[component_id])
        return result_set
