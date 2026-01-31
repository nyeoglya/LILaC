import os
import pickle
import typing as tp

import numpy as np

from processor import *

class LILaCGraph:
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.comp_map: tp.List[dict] = [] # comp id -> component
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
                self.comp_embedding_map = data.comp_embedding_map
                self.subcomp_embeddings_dump = data.subcomp_embeddings_dump
                self.subcomp_range_map = data.subcomp_range_map
                self.edge = data.edge
            print(f"Successfully loaded graph from {self.filepath}")
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
        graph.edge = [None] * N
        graph.subcomp_range_map = np.zeros((N, 2), dtype=np.int64)

        # 임베딩 차원 파악 및 총 서브컴포넌트 개수 계산 (Pre-allocation 준비)
        sample_emb = all_comps[0].embedding
        emb_dim = sample_emb.shape[0]
        total_sub_count = sum(len(c.subcomp_embeddings) for c in all_comps)

        # 미리 Numpy 배열을 할당하여 np.stack/extend의 오버헤드 제거
        graph.comp_embedding_map = np.empty((N, emb_dim), dtype=np.float32)
        graph.subcomp_embeddings_dump = np.empty((total_sub_count, emb_dim), dtype=np.float32)

        cursor = 0
        print(f"Processing {N} components and {total_sub_count} sub-embeddings...")

        # 데이터 채우기
        for comp in all_comps:
            cid = comp.id # 정렬했으므로 i와 같을 확률이 높지만 cid 사용
            
            graph.comp_map[cid] = comp.component
            graph.edge[cid] = comp.edge
            
            # 직접 할당
            graph.comp_embedding_map[cid] = comp.embedding

            sub_embs = comp.subcomp_embeddings
            k = len(sub_embs)
            
            if k > 0:
                start, end = cursor, cursor + k
                graph.subcomp_range_map[cid] = (start, end)
                # 서브컴포넌트 임베딩들을 한 번에 슬라이싱으로 할당
                graph.subcomp_embeddings_dump[start:end] = np.array(sub_embs)
                cursor = end

        # Pickle 저장 최적화
        print(f"Saving to {filepath}...")
        with open(filepath, "wb") as f:
            # HIGHEST_PROTOCOL (4 이상)을 사용해야 4GB 이상의 대용량 객체 처리가 가능하고 속도가 빠름
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True

# Graph Traverse
def find_entry(lgraph: LILaCGraph, subquery_embeddings: np.array, beam_size: int) -> tp.List[int]:
    assert subquery_embeddings.ndim == 2
    sim_matrix = subquery_embeddings @ lgraph.comp_embedding_map.T # (Q, D) @ (D, M) -> (Q, M)

    comp_scores = sim_matrix.max(axis=0)

    # top-k
    if beam_size >= len(comp_scores):
        return np.argsort(-comp_scores).tolist()

    topk = np.argpartition(-comp_scores, beam_size)[:beam_size]
    topk = topk[np.argsort(-comp_scores[topk])]

    return topk.tolist()

def one_hop(lgraph: LILaCGraph, subquery_embeddings: np.array, beam: tp.List[int], top_k: int = 5) -> tp.List[int]:
    if not beam:
        print("Warning: Input beam is empty!")
        return []

    candidates = {}
    
    # 1. 현재 beam에 있는 부모들의 점수 계산
    parent_scores = {}
    for node in beam:
        try:
            start, end = lgraph.subcomp_range_map[node]
            sim = lgraph.subcomp_embeddings_dump[start:end] @ subquery_embeddings.T
            parent_scores[node] = np.sum(np.max(sim, axis=0))
        except (KeyError, IndexError):
            continue

    # 2. 인접 노드 탐색 (lgraph.edge가 리스트인 경우)
    for parent in beam:
        p_score = parent_scores.get(parent, 0)
        
        # 리스트 범위 체크 및 해당 노드의 자식 노드 순회
        if parent < len(lgraph.edge):
            neighbors = lgraph.edge[parent] # 리스트이므로 인덱스로 접근
            
            for neighbor in neighbors:
                if neighbor not in candidates:
                    try:
                        start_n, end_n = lgraph.subcomp_range_map[neighbor]
                        n_sim = lgraph.subcomp_embeddings_dump[start_n:end_n] @ subquery_embeddings.T
                        n_score = np.sum(np.max(n_sim, axis=0))
                        
                        candidates[neighbor] = p_score + n_score
                    except (KeyError, IndexError):
                        continue

    if not candidates:
        print("No neighbors found from current beam.")
        return []

    # 3. 상위 결과 추출
    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in sorted_nodes[:top_k]]

def final_edge(lgraph: LILaCGraph, subquery_embeddings: np.array, beam: tp.List[int]) -> tp.Tuple[int, int]:
    best_overall_score = -float('inf')
    best_parent_child_pair = (-1, -1) # (parent_beam_id, child_edge_id)

    for beam_item in beam:
        # 현재 노드 점수 계산
        start_p, end_p = lgraph.subcomp_range_map[beam_item]
        parent_sim = lgraph.subcomp_embeddings_dump[start_p:end_p] @ subquery_embeddings.T
        parent_score = np.sum(np.max(parent_sim, axis=0))
        
        # 연결된 노드 중 최적 찾기
        for edge in lgraph.edge[beam_item]:
            start_c, end_c = lgraph.subcomp_range_map[edge]
            child_embs = lgraph.subcomp_embeddings_dump[start_c:end_c]
            
            # 자식 노드 점수 계산
            child_sim = child_embs @ subquery_embeddings.T
            child_score = np.sum(np.max(child_sim, axis=0))
            
            total_score = parent_score + child_score
            
            # 전체 최고점 업데이트
            if total_score > best_overall_score:
                best_overall_score = total_score
                best_parent_child_pair = (beam_item, edge)

    i1, i2 = best_parent_child_pair
    
    return lgraph.comp_map[i1], lgraph.comp_map[i2]

if __name__ == "__main__":
    LDOC_FOLDER = "/dataset/process/mmqa/"
    GRAPH_FILE_PATH = "wiki.lgraph"
    
    LILaCGraph.make_graph(LDOC_FOLDER, GRAPH_FILE_PATH)
    lilac_graph = LILaCGraph(GRAPH_FILE_PATH)
    lilac_graph.load()
    # print(sum([len(ii) for ii in lilac_graph.edge]))
