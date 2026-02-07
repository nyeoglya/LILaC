import os
import sys
import time
import json
import typing as tp

import numpy as np
from tqdm import tqdm

from graph import LILaCGraph, LILaCBeam
from query import get_subembeddings, llm_question_query
from utils.mmqa import (
    MMQAQueryEmbedding,
    mmqa_load_query_answer, mmqa_query_eval
)
from config import (
    QWEN_SERVER_URL_LIST,
    MMQA_PROCESS_IMAGE_FOLDER,
    MAX_HOP
)
from common import (
    get_embedding, get_llm_response,
    EmbeddingRequestData
)

from dataclasses import asdict
from contextlib import contextmanager

class TimeTracker:
    def __init__(self):
        self.elapsed_time = 0.0

@contextmanager
def code_timer(tracker, before, label):
    # tqdm.write(f"{before}", end="")
    start = time.time()
    yield
    delta = time.time() - start
    if tracker:
        tracker.elapsed_time += delta
        # tqdm.write(f"{label}: {tracker.elapsed_time:.3f}s (+{delta:.3f}s)")
    else:
        pass
        # tqdm.write(f"{label} ({delta:.3f}s passed)")

def process_query_list_with_cached_data(
    graph_filepath: str,
    cached_query_embedding_list: tp.List[MMQAQueryEmbedding],
    beam_size: int,
    top_k: int,
    max_hop: int,
    output_filepath: tp.Optional[str] = None
) -> tp.List[tp.Dict]:
    if output_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            pass
    
    with code_timer(None, "+ Start loading LILaCGraph... ", "Complete"):
        lilac_graph: LILaCGraph = LILaCGraph.load_graph(graph_filepath)
        if not lilac_graph:
            sys.exit(-1)
    
    graph_beam: LILaCBeam = LILaCBeam(lilac_graph)
    result_data_list: tp.List[tp.Dict] = []
    for query_data in tqdm(cached_query_embedding_list, desc="Processing Query with Cached Data..."):
        # tqdm.write(f"\nProcessing the query. qid: {query_data.qid}")
        
        with code_timer(None, "+ Get embeddings... ", "Complete"):
            query_embedding: np.ndarray = query_data.embedding
        
        with code_timer(None, "+ Get subembeddings... ", "Complete"):
            subquery_embeddings: np.ndarray = np.array(query_data.subcomponent_embedding_list)
        
        with code_timer(None, "+ Initiating beam... ", "Complete"):
            graph_beam.reset_values(beam_size, max_hop, subquery_embeddings)
        
        with code_timer(None, "+ Finding entry items from the graph... ", "Complete"):
            graph_beam.find_entry(query_embedding)
        
        with code_timer(None, "+ Processing Multi-hop on the graph... ", "Complete"):
            graph_beam.multi_hop()
        
        with code_timer(None, "+ Finding the result components... ", "Complete"):
            final_component_id_list: tp.List[int] = graph_beam.top_component_id_list(top_k=top_k)
            final_component_list: tp.List[tp.Dict] = graph_beam.top_component_list(top_k=top_k)
            final_component_uuid_set: tp.Set[str] = graph_beam.top_component_uuid_set(top_k=top_k)
        final_component_uuid_list = list(final_component_uuid_set)

        # tqdm.write(f"Final Components: {final_component_id_list}")
        
        result_data: tp.Dict[str, tp.Any] = {
            'qid': query_data.qid,
            'final_component_id_list': final_component_id_list,
            'final_component_list': final_component_list,
            'final_component_uuid_list': final_component_uuid_list
        }
        
        result_data_list.append(result_data)
        
        if output_filepath:
            with open(output_filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
    
    return result_data_list

def process_query_list_retrieval(
    graph_filepath: str,
    retrieval_caching_filepath: str,
    embedding_server_url: str,
    query_list: list[str],
    beam_size: int,
    max_hop: int,
    top_k: int
) -> tp.List[tp.List[tp.Dict]]:
    time_tracker = TimeTracker()
    with code_timer(None, "+ Start loading LILaCGraph... ", "Complete"):
        lilac_graph = LILaCGraph.load_graph(graph_filepath)
        if not lilac_graph:
            sys.exit(-1)
    
    result_comps_list: tp.List[tp.List[dict]] = []
    elapsed_time_list: tp.List[float] = []

    with open(retrieval_caching_filepath, 'a', encoding='utf-8') as retrieval_caching_file:
        for query in tqdm(query_list):
            time_tracker.elapsed_time = 0.0
            tqdm.write(f"\nProcessing the query: {query}")
            
            with code_timer(time_tracker, "+ Get embedding... ", "Complete"):
                embedding = get_embedding(embedding_server_url, EmbeddingRequestData(query))
            
            with code_timer(time_tracker, "+ Get subquery embeddings... ", "Complete"):
                subquery_embeddings = get_subembeddings(embedding_server_url, query)
            
            beam = LILaCBeam(lilac_graph)
            
            with code_timer(None, "+ Initiating beam... ", "Complete"):
                beam.reset_values(beam_size, max_hop, subquery_embeddings)
            
            with code_timer(time_tracker, "+ Finding entry items from the graph... ", "Complete"):
                beam.find_entry(embedding)
            
            hop_count = 0
            while hop_count < max_hop:
                with code_timer(time_tracker, f"+ Trying one hop (step {hop_count+1})... ", "Complete"):
                    changed = beam.one_hop()
                
                if not changed:
                    tqdm.write(f"+ Convergence reached at {hop_count+1} hops.")
                    break
                hop_count += 1

            with code_timer(time_tracker, "+ Finding the final components... ", "Complete"):
                final_comp_ids = beam.top_component_id_list(top_k=top_k)
                comps = beam.top_component_list(top_k=top_k)

            tqdm.write(f"Final Components: {final_comp_ids}. Total elapsed time: {time_tracker.elapsed_time:.3f}s")
            
            record = {
                'query': query,
                'comp_ids': final_comp_ids,
                'comps': comps,
                'elapsed_time': time_tracker.elapsed_time
            }
            
            retrieval_caching_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            retrieval_caching_file.flush()

            result_comps_list.append(comps)
            elapsed_time_list.append(time_tracker.elapsed_time)
    
    print("\n", "+ Total Result\n")
    print(f"Total Elapsed Time: {sum(elapsed_time_list):.3f}s\n\n")
    
    return result_comps_list

# def main_process(query_list: list[str], graph_filepath: str, temp_graph_filepath: str, temp_llm_filepath: str, image_folderpath: str):
#     query_list = sorted(query_list)
#     result_comps_list, result_docs_list = process_query_list_retrieval(graph_filepath, query_list, temp_graph_filepath)

#     llm_cache = {}
#     if os.path.exists(temp_llm_filepath):
#         with open(temp_llm_filepath, "r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     data = json.loads(line.strip())
#                     llm_cache[data['query']] = data['answer']
#                 except: continue
#         print(f"이미 완료된 {len(llm_cache)}개의 LLM 응답을 로드했습니다.")

#     llm_response_list = []
    
#     # LLM 처리 루프
#     with open(temp_llm_filepath, "a", encoding="utf-8") as f_out:
#         for ind in range(len(query_list)):
#             q = query_list[ind]
            
#             if q in llm_cache:
#                 print(f"[{ind+1}/{len(query_list)}] Skipping (already answered): {q[:30]}...")
#                 llm_response_list.append(llm_cache[q])
#                 continue

#             print(f"[{ind+1}/{len(query_list)}] Processing LLM query: {q[:50]}...")
#             final_query, img_paths = llm_question_query(
#                 q, image_folderpath, result_docs_list[ind], result_comps_list[ind]
#             )
            
#             try:
#                 llm_response = get_llm_response(QWEN_SERVER_URL_LIST[0], final_query, img_paths)
#             except Exception as e:
#                 print(f"!!! Error at query {ind}: {e}")
#                 llm_response = "ERROR: Generation Failed"

#             llm_response_list.append(llm_response)
#             print(f"=> LLM Augmented Answer: {llm_response[:100]}\n")
            
#             f_out.write(json.dumps({"query": q, "answer": llm_response}, ensure_ascii=False) + "\n")
#             f_out.flush()
    
#     return llm_response_list, result_comps_list
