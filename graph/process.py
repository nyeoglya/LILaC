import os
import sys
import time
import json
import typing as tp
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from graph import LILaCGraph, LILaCBeam
from query import get_subembeddings, LLMQueryGenerator
from utils.mmqa import MMQAQueryEmbedding, MMQAQueryAnswer
from common import (
    get_embedding, get_llm_response,
    EmbeddingRequestData
)

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
        assert not os.path.exists(output_filepath)
    
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

def process_llm_answer(
    query_retrieval_component_list: tp.List[tp.Tuple[MMQAQueryAnswer, tp.List[tp.Dict]]],
    llm_server_url: str,
    llm_answer_filepath: str,
    failed_filepath: str,
    image_folderpath: str
) -> tp.List[str]:
    assert not os.path.exists(llm_answer_filepath)
    
    llm_query_generator = LLMQueryGenerator(image_folderpath)
    llm_response_list = []
    with open(llm_answer_filepath, "w", encoding="utf-8") as llm_answer_file,\
        open(failed_filepath, "w", encoding="utf-8") as failed_file:
        for query_answer, retrieval_component in tqdm(query_retrieval_component_list, desc="Processing LLM query..."):
            final_query, image_filepath_list = llm_query_generator.llm_question_query(query_answer.question, retrieval_component)
            try:
                llm_response = get_llm_response(llm_server_url, final_query, image_filepath_list)
            except Exception as e:
                tqdm.write(f"Error on query: {query_answer.qid}. Error: {e}")
                failed_file.write(f"Error on query: {query_answer.qid}. Error: {e}")
                llm_response = "[Generation Failed]"

            llm_response_list.append(llm_response)
            llm_answer_file.write(json.dumps({"qid": query_answer.qid, "answer": llm_response}, ensure_ascii=False) + "\n")
            llm_answer_file.flush()
    return llm_response_list

def multiprocess_llm_answer(
    query_retrieval_component_list: tp.List[tp.Tuple[MMQAQueryAnswer, tp.List[tp.Dict]]],
    llm_server_url_list: tp.List[str],
    llm_answer_filepath: str,
    failed_filepath: str,
    image_folderpath: str,
) -> tp.List[str]:
    assert not os.path.exists(llm_answer_filepath)
    
    server_cycle = cycle(llm_server_url_list)
    results_map: tp.Dict[str, str] = {}
    
    def fetch_answer(item):
        query_answer, retrieval_component = item
        local_generator = LLMQueryGenerator(image_folderpath)
        server_url = next(server_cycle)
        
        final_query, image_filepath_list = local_generator.llm_question_query(
            query_answer.question, retrieval_component
        )
        
        try:
            response = get_llm_response(server_url, final_query, image_filepath_list, max_tokens=4096)
            return query_answer.qid, response, None
        except Exception as e:
            return query_answer.qid, "[Generation Failed]", f"Error on {query_answer.qid}: {str(e)}"

    with open(llm_answer_filepath, "w", encoding="utf-8") as llm_answer_file, \
         open(failed_filepath, "w", encoding="utf-8") as failed_file:
        
        with ThreadPoolExecutor(max_workers=len(llm_server_url_list)) as executor:
            future_list = [executor.submit(fetch_answer, item) for item in query_retrieval_component_list]
            
            for future in tqdm(as_completed(future_list), total=len(future_list), desc="Parallel Query Processing..."):
                qid, response, error_msg = future.result()
                
                results_map[qid] = response
                
                if error_msg:
                    tqdm.write(error_msg)
                    failed_file.write(error_msg + "\n")
                
                llm_answer_file.write(json.dumps({"qid": qid, "answer": response}, ensure_ascii=False) + "\n")
                llm_answer_file.flush()

    return [results_map[qa.qid] for qa, _ in query_retrieval_component_list]
