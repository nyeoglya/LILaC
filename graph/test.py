import os
import typing as tp

import numpy as np
from tqdm import tqdm

from graph import LILaCDocument, ProcessedComponent
from utils.mmqa import MMQAQueryAnswer, MMQAQueryEmbedding

def mmqa_graph_retrieval_test(query_answer_list: tp.List[MMQAQueryAnswer], retrieval_result_list: tp.List[tp.Dict]):
    query_answer_dict: tp.Dict[str, MMQAQueryAnswer] = {query_answer_data.qid: query_answer_data for query_answer_data in query_answer_list}
    score = 0
    perfect_score = 0
    max_count = 0
    for retrieval_result in tqdm(retrieval_result_list, desc="Evaluating LILaC Retrieval Score..."):
        query_id = retrieval_result["qid"]
        final_component_uuid_list = retrieval_result["final_component_uuid_list"]
        query_answer_data = query_answer_dict[query_id]
        
        ground_truth_uuid_set = set(query_answer_data.supporting_context_id_list)
        predicted_component_uuid_set = set(final_component_uuid_list)

        if ground_truth_uuid_set.issubset(predicted_component_uuid_set):
            perfect_score += 1
        for ground_truth_uuid in ground_truth_uuid_set:
            if ground_truth_uuid in predicted_component_uuid_set:
                score += 1
        max_count += 1
    
    print(f"Match Exist Count: {score} | Perfect Match Count: {perfect_score} | Total Count: {max_count}")
    print(f"Perfect Match Score: {perfect_score / max_count}")
    print(f"Match Exist Score: {score / max_count}")

def mmqa_embed_knn_test(query_answer_list: tp.List[MMQAQueryAnswer], query_embedding_list: tp.List[MMQAQueryEmbedding], ldoc_folderpath: str, top_k: int = 9):
    assert len(query_answer_list) == len(query_embedding_list)
    
    ldoc_embedding_list: tp.List[np.ndarray] = []
    ldoc_qids_list: tp.List[tp.List[str]] = []
    for ldoc_filename in os.listdir(ldoc_folderpath):
        ldoc_filepath = os.path.join(ldoc_folderpath, ldoc_filename)
        ldoc = LILaCDocument.load_from_path(ldoc_filepath)
        if not ldoc:
            continue
        for processed_component in ldoc.processed_components:
            ldoc_embedding_list.append(processed_component.component_embedding)
            ldoc_qids_list.append(processed_component.component_uuid)
    ldoc_embedding_dump: np.ndarray = np.array(ldoc_embedding_list)
    
    query_answer_dict: tp.Dict[str, MMQAQueryAnswer] = {query_answer_data.qid: query_answer_data for query_answer_data in query_answer_list}
    score = 0
    perfect_score = 0
    max_count = 0
    for query_embedding_data in tqdm(query_embedding_list, desc="Evaluating MM-Embed Score..."):
        query_id = query_embedding_data.qid
        query_answer_data = query_answer_dict[query_id]
        
        similarity_scores = ldoc_embedding_dump @ query_embedding_data.embedding
        top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        retrieved_qids = [ldoc_qids_list[i] for i in top_k_indices]
        flattened_retrieved_qids = [item for sublist in retrieved_qids for item in sublist]
        
        ground_truth_uuid_set = set(query_answer_data.supporting_context_id_list)
        predicted_component_uuid_set = set(flattened_retrieved_qids)

        if ground_truth_uuid_set.issubset(predicted_component_uuid_set):
            perfect_score += 1
        for ground_truth_uuid in ground_truth_uuid_set:
            if ground_truth_uuid in predicted_component_uuid_set:
                score += 1
        max_count += 1
    
    print(f"Match Exist Count: {score} | Perfect Match Count: {perfect_score} | Total Count: {max_count}")
    print(f"Perfect Match Score: {perfect_score / max_count}")
    print(f"Match Exist Score: {score / max_count}")
    
def mmqa_verify_ldoc(ldoc_folderpath: str): # TODO: temp function
    collision_samples = []
    i = 0
    for ldoc_filename in os.listdir(ldoc_folderpath):
        ldoc_filepath = os.path.join(ldoc_folderpath, ldoc_filename)
        ldoc = LILaCDocument.load_from_path(ldoc_filepath)
        if not ldoc:
            continue
        for processed_component in ldoc.processed_components:
            if processed_component.component_uuid:
                i += len(processed_component.component_uuid)
                if len(processed_component.component_uuid) > 1:
                    collision_samples.append({
                        'type': processed_component.original_component['type'],
                        'uuids': processed_component.component_uuid,
                        'content_preview': str(processed_component.original_component.get('paragraph', ''))[:30]
                    })
            # processed_component.component_embedding

    for s in collision_samples:
        print(f"[{s['type']}] 중복 ID: {s['uuids']} | 내용: {s['content_preview']}...")
