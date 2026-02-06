import os
import pickle
import typing as tp

import numpy as np

from processor import LILaCDocument
from common import get_embedding, EmbeddingRequestData

class LILaCGraph:
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.comp_map: tp.List[dict] = [] # comp id -> component
        self.comp_doc_map: tp.List[str] = [] # comp id -> doc title
        self.comp_embedding_map: np.ndarray = np.array([]) # comp id -> comp embedding
        
        self.subcomp_embeddings_dump: np.ndarray = np.array([]) # comp id -> array slicing(=subcomponent_embedding)
        self.subcomp_range_map: np.ndarray = np.array([]) # comp id -> (start, end) tuple
        self.edge: tp.List[tp.Set[int]] = [] # comp id -> set(comp id)

    # Manage Files
    def load(self) -> bool:
        if not os.path.exists(self.filepath):
            print(f"Error: {self.filepath} not found.")
            return False
        
        try:
            with open(self.filepath, "rb") as f:
                data = pickle.load(f)
                self.comp_map = data.comp_map
                self.comp_doc_map = data.comp_doc_map
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
        assert os.path.exists(remapped_ldoc_path)

        fnames = [f for f in os.listdir(remapped_ldoc_path) if f.endswith(".ldoc")]
        ldocs_with_none: tp.List[tp.Optional[LILaCDocument]] = [LILaCDocument.load_from_path(os.path.join(remapped_ldoc_path, f)) for f in fnames]
        ldocs: tp.List[LILaCDocument] = [doc for doc in ldocs_with_none if doc is not None]

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
        graph.edge = [set() for _ in range(N)]
        graph.subcomp_range_map = np.zeros((N, 2), dtype=np.int64)

        # 임베딩 차원 파악 및 총 서브 컴포넌트 개수 계산
        sample_emb = all_comps[0].embedding
        emb_dim = sample_emb.shape[0]
        total_sub_count = sum(len(c.subcomp_embeddings) for c in all_comps)

        graph.comp_embedding_map = np.empty((N, emb_dim), dtype=np.float32)
        graph.subcomp_embeddings_dump = np.empty((total_sub_count, emb_dim), dtype=np.float32)

        cursor = 0
        print(f"Processing {N} components and {total_sub_count} sub-embeddings...")

        # 데이터 채우기
        for ldoc in ldocs:
            doc_comp_ids = [comp.id for comp in ldoc.processed_components]
            
            for comp in ldoc.processed_components:
                cid = comp.id
                
                # 기본 정보 할당
                graph.comp_map[cid] = comp.component
                graph.comp_doc_map[cid] = ldoc.doc_title
                graph.comp_embedding_map[cid] = comp.embedding

                # 같은 문서 내 모든 컴포넌트와 양방향 연결
                for other_id in doc_comp_ids:
                    graph.edge[cid].add(other_id)
                    graph.edge[other_id].add(cid) # 역방향 추가
                    
                # 기존 comp.edge에 정의된 관계를 양방향 연결
                if comp.edge:
                    for target_id in comp.edge:
                        graph.edge[cid].add(target_id)
                        graph.edge[target_id].add(cid) # 역방향 추가
                
                # 임베딩 처리
                sub_embs = comp.subcomp_embeddings
                k = len(sub_embs)
                if k > 0:
                    start, end = cursor, cursor + k
                    graph.subcomp_range_map[cid] = (start, end)
                    graph.subcomp_embeddings_dump[start:end] = np.array(sub_embs)
                    cursor = end

        # Pickle 저장 최적화
        print(f"Saving to {filepath}...")
        with open(filepath, "wb") as f: # TODO: pickle 대신에 tensor pt 써보기
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL) # HIGHEST_PROTOCOL: 4GB 이상의 대용량 객체 처리에 유용

        return True

class LILaCBeam:
    def __init__(self, lgraph: LILaCGraph, query_embedding: np.ndarray, subquery_embeddings: np.ndarray, beam_size: int = 5) -> None:
        assert subquery_embeddings.ndim == 2
        assert query_embedding.ndim == 1
        
        self.query_embedding: np.array = query_embedding
        self.subquery_embeddings: np.array = subquery_embeddings
        self.beam_size = beam_size
        self.lilac_graph = lgraph
        self.beam: tp.List[tp.Tuple[int, int]] = []

    # Graph Traverse
    def find_entry(self) -> bool:
        comp_scores = self.query_embedding @ self.lilac_graph.comp_embedding_map.T # (1, D) @ (D, M) -> (1, M)

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
                if (c1, c2) in candidate_edges:
                    continue

                score_edge = self.calculate_score(c1, c2)
                score_c1 = self.calculate_score(c1)
                score_c2 = self.calculate_score(c2)

                # one sided match
                if score_edge <= max(score_c1, score_c2) + 1e-5:
                    winner = c1 if score_c1 >= score_c2 else c2
                    if (winner, winner) not in candidate_edges:
                        candidate_edges[(winner, winner)] = max(score_c1, score_c2)
                else:
                    candidate_edges[(c1, c2)] = score_edge

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

class LILaCBeamV2:
    def __init__(self, lgraph: 'LILaCGraph', query_embedding: np.ndarray, subquery_embeddings: np.ndarray, beam_size: int = 5) -> None:
        self.query_embedding = query_embedding
        self.subquery_embeddings = subquery_embeddings
        self.beam_size = beam_size
        self.lilac_graph = lgraph
        
        # Beam에는 (comp1, comp2) 튜플 형태의 엣지 혹은 (comp, comp) 형태의 단일 노드가 유지됨
        self.beam: tp.List[tp.Tuple[int, int]] = []

    def _get_top_relevancy(self, comp_id: int) -> float:
        """Top-level 임베딩(comp_embedding_map)과 원본 쿼리 간의 내적 점수를 반환합니다."""
        comp_vec = self.lilac_graph.comp_embedding_map[comp_id]
        return float(np.dot(comp_vec, self.query_embedding))

    def find_entry(self) -> bool:
        """Low-level(서브 컴포넌트) 전체에서 k-NN 수행 후 부모 ID로 변환하여 진입합니다."""
        
        # 1. 모든 서브 컴포넌트 임베딩과 질문 간의 유사도 계산
        # lilac_graph.subcomp_embeddings_dump: (Total_Sub_Count, D)
        low_scores = self.lilac_graph.subcomp_embeddings_dump @ self.query_embedding.T
        
        # 2. 상위 2048개 하위 인덱스 추출
        top_k_low_indices = np.argsort(-low_scores)[:2048]
        
        # 3. 하위 인덱스가 어느 부모(comp_id)에 속하는지 매핑
        # subcomp_range_map을 순회하며 부모 ID 탐색 (성능을 위해 역매핑 테이블이 있으면 좋으나, 직접 계산)
        seen_parents = []
        for low_idx in top_k_low_indices:
            # low_idx가 어느 (start, end) 구간에 있는지 확인
            parent_id = -1
            # subcomp_range_map: (N, 2)
            starts = self.lilac_graph.subcomp_range_map[:, 0]
            ends = self.lilac_graph.subcomp_range_map[:, 1]
            
            # low_idx가 포함된 행 찾기
            matches = np.where((starts <= low_idx) & (low_idx < ends))[0]
            if len(matches) > 0:
                parent_id = matches[0]
            
            if parent_id != -1 and parent_id not in seen_parents:
                seen_parents.append(parent_id)
            
            if len(seen_parents) >= self.beam_size:
                break
        
        self.beam = [(p, p) for p in seen_parents]
        return len(self.beam) > 0

    def calculate_score(self, comp1: int, comp2: tp.Optional[int] = None) -> float:
        """논문 수식 (6)의 Late Interaction 스코어링입니다."""
        start1, end1 = self.lilac_graph.subcomp_range_map[comp1]
        emb1 = self.lilac_graph.subcomp_embeddings_dump[start1:end1]
        
        if comp2 is not None and comp1 != comp2:
            start2, end2 = self.lilac_graph.subcomp_range_map[comp2]
            emb2 = self.lilac_graph.subcomp_embeddings_dump[start2:end2]
            combined_subembs = np.concatenate((emb1, emb2), axis=0)
        else:
            combined_subembs = emb1

        if combined_subembs.size == 0: return -1.0

        # 각 서브쿼리(Q)에 대해 하위 컴포넌트(M) 중 최대 유사도를 찾아 합산
        # (Q, D) @ (D, M) -> (Q, M)
        sim_matrix = self.subquery_embeddings @ combined_subembs.T 
        return float(np.sum(np.max(sim_matrix, axis=1)))

    def one_hop(self) -> bool:
        """현재 빔의 노드들에서 인접 노드로 확장하고 Late Interaction으로 재평가합니다."""
        if not self.beam: return False

        current_nodes = set(c for edge in self.beam for c in edge)
        candidate_edges = {} 

        for comp in current_nodes:
            neighbors = self.lilac_graph.edge[comp] # set(comp_id)
            
            # 1. 고립 노드 처리 (더미 엣지 개념)
            if not neighbors:
                candidate_edges[frozenset([comp])] = self.calculate_score(comp)
                continue

            # 2. 인접 노드 확장 및 엣지 스코어링
            for neighbor in neighbors:
                edge_key = frozenset([comp, neighbor])
                if edge_key in candidate_edges: continue
                
                score_edge = self.calculate_score(comp, neighbor)
                score_c1 = self.calculate_score(comp)
                score_c2 = self.calculate_score(neighbor)

                # One-sided match logic (논문 4.2.2)
                if score_edge <= max(score_c1, score_c2) + 1e-5:
                    winner = comp if score_c1 >= score_c2 else neighbor
                    candidate_edges[frozenset([winner])] = max(score_c1, score_c2)
                else:
                    candidate_edges[edge_key] = score_edge

        # 점수 기준 정렬 및 상위 beam_size개 선택
        sorted_results = sorted(candidate_edges.items(), key=lambda x: x[1], reverse=True)
        top_selections = sorted_results[:self.beam_size]

        # In-edge Re-ranking: 엣지 내에서 질문과 더 관련 있는 노드를 0번에 배치
        new_beam = []
        for nodes_set, _ in top_selections:
            nodes = list(nodes_set)
            if len(nodes) > 1:
                # Top-level 임베딩 유사도 기준으로 정렬
                nodes.sort(key=lambda n: self._get_top_relevancy(n), reverse=True)
                new_beam.append((nodes[0], nodes[1]))
            else:
                new_beam.append((nodes[0], nodes[0]))

        prev_nodes = current_nodes
        self.beam = new_beam
        curr_nodes = set(c for edge in self.beam for c in edge)
        
        return prev_nodes != curr_nodes

    def top_comp_ids(self, top_k: int) -> tp.List[int]:
        if not self.beam: raise IndexError("Beam is empty.")
        
        ordered_nodes = []
        seen = set()
        for edge in self.beam:
            for node_id in edge:
                # 여기서 int()로 감싸서 NumPy 타입을 파이썬 기본 타입으로 변환합니다.
                python_node_id = int(node_id) 
                
                if python_node_id not in seen:
                    seen.add(python_node_id)
                    ordered_nodes.append(python_node_id)
                
                if len(ordered_nodes) >= top_k: 
                    return ordered_nodes
        return ordered_nodes

    def top_comps(self, top_k: int) -> tp.List[dict]:
        return [self.lilac_graph.comp_map[nid] for nid in self.top_comp_ids(top_k)]

    def top_doc_titles(self, top_k: int) -> tp.List[str]:
        return [self.lilac_graph.comp_doc_map[nid] for nid in self.top_comp_ids(top_k)]

if __name__ == "__main__":
    LILaCGraph.make_graph(LDOC_FOLDER, GRAPH_FILE_PATH)
    lilac_graph = LILaCGraph(GRAPH_FILE_PATH)
    lilac_graph.load()
    # print(sum([len(ii) for ii in lilac_graph.edge]))

    for i in range(100):
        aaa = lilac_graph.comp_map[i]
        if not aaa["type"] == "paragraph":
            continue
        text = aaa["paragraph"]
        emb = lilac_graph.comp_embedding_map[i]
        q_emb = get_embedding(EmbeddingRequestData(text, ""))
        sim = np.dot(emb, q_emb.T)
        if sim < 0.9:
            print(f"Index {i} broken! Sim: {sim}")
            print(f"{lilac_graph.comp_map[i]}")
