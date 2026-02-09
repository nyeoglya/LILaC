import os
import typing as tp

import numpy as np
from tqdm import tqdm

from graph import LILaCDocument, ProcessedComponent
from utils.mmqa import MMQAQueryAnswer, MMQAQueryEmbedding
from .base import normalize_answer

def mmqa_graph_retrieval_mrr_test(query_answer_list: tp.List[MMQAQueryAnswer], retrieval_result_list: tp.List[tp.Dict]):
    query_answer_dict: tp.Dict[str, MMQAQueryAnswer] = {query_answer_data.qid: query_answer_data for query_answer_data in query_answer_list}
    
    score = 0
    total_mrr = 0.0
    max_count = 0
    for retrieval_result in tqdm(retrieval_result_list, desc="Evaluating Retrieval MRR Score..."):
        query_id = retrieval_result["qid"]
        final_component_uuid_list = retrieval_result["final_component_uuid_list"]
        query_answer_data = query_answer_dict[query_id]
        ground_truth_uuid_set = set(query_answer_data.supporting_context_id_list)

        found_rank = 0
        for i, pred_uuid in enumerate(final_component_uuid_list):
            if pred_uuid in ground_truth_uuid_set:
                found_rank = i + 1
                break
        if found_rank > 0:
            score += 1
            total_mrr += (1.0 / found_rank)
        
        max_count += 1
    
    mrr_score = total_mrr / max_count if max_count > 0 else 0
    
    print(f"MRR: {mrr_score:.4f}")

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
                break
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
                break
        max_count += 1
    
    print(f"Match Exist Count: {score} | Perfect Match Count: {perfect_score} | Total Count: {max_count}")
    print(f"Perfect Match Score: {perfect_score / max_count}")
    print(f"Match Exist Score: {score / max_count}")

def mmqa_query_eval(llm_answer_list_map: tp.Dict[str, tp.List[str]], ground_truth_list_map: tp.Dict[str, tp.List[str]]):
    total_em = 0
    total_f1 = 0.0
    max_count = 0
    
    f1_scores = []
    for qid in tqdm(llm_answer_list_map, desc="Calculating EM & F1..."):
        if qid not in ground_truth_list_map:
            continue
        
        llm_prediction_set = {normalize_answer(str(ans)) for ans in llm_answer_list_map[qid] if normalize_answer(str(ans))}
        ground_truth_set = {normalize_answer(str(gt)) for gt in ground_truth_list_map[qid] if normalize_answer(str(gt))}
        if not ground_truth_set: continue

        max_count += 1
        if llm_prediction_set == ground_truth_set: total_em += 1
        if not llm_prediction_set: current_f1 = 0.0
        else:
            common = llm_prediction_set.intersection(ground_truth_set)
            precision = len(common) / len(llm_prediction_set)
            recall = len(common) / len(ground_truth_set)
            if (precision + recall) > 0: current_f1 = (2 * precision * recall) / (precision + recall)
            else: current_f1 = 0.0
        total_f1 += current_f1
        f1_scores.append(current_f1)

    final_em = (total_em / max_count) * 100 if max_count > 0 else 0
    final_f1 = (total_f1 / max_count) * 100 if max_count > 0 else 0

    print(f"Evaluation Results (N={max_count})")
    print(f"Exact Match: {final_em:.2f}%")
    print(f"F1 Score: {final_f1:.2f}%")
