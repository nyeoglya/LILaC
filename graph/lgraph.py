import os
import pickle
import typing as tp

import numpy as np

from processor import *

class LILaCGraph:
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.comp_map: tp.List[dict] = [] # comp id -> component
        self.comp_doc_map: tp.List[str] = [] # comp id -> doc title
        self.comp_embedding_map: np.array = np.array([]) # comp id -> comp embedding
        
        self.subcomp_embeddings_dump: np.array = np.array([]) # comp id -> array slicing(=subcomponent_embedding)
        self.subcomp_range_map: np.array = np.array([]) # comp id -> (start, end) tuple
        self.edge: tp.List[tp.List[int]] = [] # comp id -> list(comp id)

    # Manage Files
    def load(self) -> bool:
        if not os.path.exists(self.filepath):
            print(f"Error: {self.filepath} not found.")
            return False
        
        try:
            with open(self.filepath, "rb") as f:
                # 저장된 객체를 불러와서 현재 인스턴스의 속성들에 덮어씌움
                data = pickle.load(f)
                self.comp_map = data.comp_map
                self.comp_doc_map = data.comp_doc_map # 필드 추가
                self.comp_embedding_map = data.comp_embedding_map
                self.subcomp_embeddings_dump = data.subcomp_embeddings_dump
                self.subcomp_range_map = data.subcomp_range_map
                self.edge = data.edge
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False
    
    @staticmethod
    def make_graph(remapped_ldoc_path: str, filepath: str) -> bool:
        import os
        import pickle
        import numpy as np

        if not os.path.exists(remapped_ldoc_path):
            return False

        # 문서 로드 (리스트 컴프리헨션으로 속도 향상)
        fnames = [f for f in os.listdir(remapped_ldoc_path) if f.endswith(".ldoc.remapped")]
        ldocs = [LILaCDocument.load(os.path.join(remapped_ldoc_path, f)) for f in fnames]
        ldocs = [doc for doc in ldocs if doc is not None]

        # 컴포넌트 수집 및 정렬
        all_comps = []
        for ldoc in ldocs:
            all_comps.extend(ldoc.processed_components)
        all_comps.sort(key=lambda c: c.id)
        
        N = len(all_comps)
        if N == 0: return False

        graph = LILaCGraph(filepath)
        graph.comp_map = [None] * N
        graph.comp_doc_map = [None] * N
        graph.edge = [None] * N
        graph.subcomp_range_map = np.zeros((N, 2), dtype=np.int64)

        # 임베딩 차원 파악 및 총 서브 컴포넌트 개수 계산 (Pre-allocation 준비)
        sample_emb = all_comps[0].embedding
        emb_dim = sample_emb.shape[0]
        total_sub_count = sum(len(c.subcomp_embeddings) for c in all_comps)

        # 미리 Numpy 배열을 할당하여 np.stack/extend의 오버헤드 제거
        graph.comp_embedding_map = np.empty((N, emb_dim), dtype=np.float32)
        graph.subcomp_embeddings_dump = np.empty((total_sub_count, emb_dim), dtype=np.float32)

        cursor = 0
        print(f"Processing {N} components and {total_sub_count} sub-embeddings...")

        # 데이터 채우기
        for ldoc in ldocs:
            # 현재 문서에 속한 모든 컴포넌트의 ID 목록 추출
            doc_comp_ids = [comp.id for comp in ldoc.processed_components]
            
            for comp in ldoc.processed_components:
                cid = comp.id
                
                graph.comp_map[cid] = comp.component
                graph.comp_doc_map[cid] = ldoc.doc_title
                
                # 엣지 생성 로직
                edges = set(comp.edge) if comp.edge else set()
                edges.add(cid)
                edges.update(doc_comp_ids)
                graph.edge[cid] = list(edges)
                
                # 임베딩 할당
                graph.comp_embedding_map[cid] = comp.embedding

                sub_embs = comp.subcomp_embeddings
                k = len(sub_embs)
                if k > 0:
                    start, end = cursor, cursor + k
                    graph.subcomp_range_map[cid] = (start, end)
                    graph.subcomp_embeddings_dump[start:end] = np.array(sub_embs)
                    cursor = end

        # Pickle 저장 최적화
        print(f"Saving to {filepath}...")
        with open(filepath, "wb") as f:
            # HIGHEST_PROTOCOL: 4GB 이상의 대용량 객체 처리에 유용
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True

class LILaCBeam:
    def __init__(self, lgraph: LILaCGraph, subquery_embeddings: np.array, beam_size: int = 5) -> None:
        assert subquery_embeddings.ndim == 2
        
        self.subquery_embeddings: np.array = subquery_embeddings
        self.beam_size = beam_size
        self.lilac_graph = lgraph
        self.beam: tp.List[tp.Tuple[int, int]] = []

    # Graph Traverse
    def find_entry(self) -> bool:
        sim_matrix = self.subquery_embeddings @ self.lilac_graph.comp_embedding_map.T # (Q, D) @ (D, M) -> (Q, M)

        comp_scores = sim_matrix.max(axis=0)

        # top beam size
        beam_entry = []
        if self.beam_size >= len(comp_scores):
            beam_entry = np.argsort(-comp_scores).tolist()
        else:
            topk = np.argpartition(-comp_scores, self.beam_size)[:self.beam_size]
            topk = topk[np.argsort(-comp_scores[topk])]
            beam_entry = topk.tolist()
        self.beam = [(cand, cand) for cand in beam_entry]
        return True

    def calculate_score(self, comp1, comp2=None):
        start1, end1 = self.lilac_graph.subcomp_range_map[comp1]
        sim1 = self.subquery_embeddings @ self.lilac_graph.subcomp_embeddings_dump[start1:end1].T # (Q, D) @ (D, M)
        if comp2:
            start2, end2 = self.lilac_graph.subcomp_range_map[comp2]
            sim2 = self.subquery_embeddings @ self.lilac_graph.subcomp_embeddings_dump[start2:end2].T # (Q, D) @ (D, M)
            return np.sum(np.max(np.concatenate((sim1, sim2), axis=1), axis=1))
        else:
            return np.sum(np.max(sim1, axis=1))

    def one_hop(self) -> bool:
        if not self.beam:
            return False

        candidate_comps = set()
        for beam_elem in self.beam:
            candidate_comps.add(beam_elem[0])
            candidate_comps.add(beam_elem[1])

        prev_beam_nodes = set(c for edge in self.beam for c in edge)
        candidate_edges = dict() # (comp1, comp2) -> score dict. (comp1 > comp2)        
        for comp in candidate_comps:
            neighbor_comps = self.lilac_graph.edge[comp]
            if not neighbor_comps:
                if (comp, comp) not in candidate_edges:
                    candidate_edges[(comp, comp)] = self.calculate_score(comp)
                continue
                
            for neighbor_comp in neighbor_comps:
                c1, c2 = (comp, neighbor_comp) if comp > neighbor_comp else (neighbor_comp, comp)
                if (c1, c2) not in candidate_edges:
                    candidate_edges[(c1, c2)] = self.calculate_score(c1, c2)

        # 부모와 자식을 통틀어 가장 점수가 높은 beam_size개 edge 추출
        sorted_nodes = sorted(candidate_edges.items(), key=lambda x: x[1], reverse=True)
        self.beam = [node for node, score in sorted_nodes[:self.beam_size]]
        curr_beam_nodes = set(c for edge in self.beam for c in edge)
        return prev_beam_nodes != curr_beam_nodes

    def top_comp_ids(self, top_k: int) -> tp.List[int]:
        if not self.beam:
            raise IndexError("Beam is empty. Run find_entry or one_hop first.")
        
        all_nodes = [c for edge in self.beam for c in edge]
        unique_nodes = list(dict.fromkeys(all_nodes))
        return unique_nodes[:top_k]
    
    def top_comps(self, top_k: int) -> tp.List[dict]:
        result_comps = []
        unique_nodes = self.top_comp_ids(top_k)
        for unique_node in unique_nodes:
            result_comps.append(self.lilac_graph.comp_map[unique_node])
        return result_comps

    def top_doc_titles(self, top_k: int) -> tp.List[dict]:
        result_docs = []
        unique_nodes = self.top_comp_ids(top_k)
        for unique_node in unique_nodes:
            result_docs.append(self.lilac_graph.comp_doc_map[unique_node])
        return result_docs

if __name__ == "__main__":
    LDOC_FOLDER = "/dataset/process/mmqa/"
    GRAPH_FILE_PATH = "wiki.lgraph"
    
    # LILaCGraph.make_graph(LDOC_FOLDER, GRAPH_FILE_PATH)
    lilac_graph = LILaCGraph(GRAPH_FILE_PATH)
    lilac_graph.load()
    print(sum([len(ii) for ii in lilac_graph.edge]))
