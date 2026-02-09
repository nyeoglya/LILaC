import json
import pickle
import typing as tp

from embed import SequentialComponentEmbedder
from graph import LILaCGraph, LILaCDocument, ProcessedComponent
from utils.mmqa import (
    MMQAQueryEmbedding, MMQAQueryAnswer,
    mmqa_load_query_answer, mmqa_cache_query_process, mmqa_load_cached_query_data, load_retrieval_result_map
)
from evaluation.base import extract_answer_list_from_f_call
from evaluation.mmqa import mmqa_embed_knn_test, mmqa_graph_retrieval_test, mmqa_query_eval, mmqa_graph_retrieval_mrr_test
from remap import LILaCDocMMQALabeler
from process import process_query_list_with_cached_data, multiprocess_llm_answer
from config import (
    MMQA_PATH, MMQA_LDOC_FOLDER, MMQA_PROCESS_IMAGE_FOLDER,
    MMQA_PARSE_JSON_FOLDER,
    MMQA_IMAGE_DESCRIPTION_INFO_FILE, MMQA_OBJECT_DETECT_INFO_FILE,
    MMQA_IMAGE_EMBEDDING_FOR_LABELING_FILE, MMQA_IMAGE_REFERENCE_EMBEDDING_FOR_LABELING_FILE, MMQA_LABELED_LDOC_FOLDER,
    QWEN_SERVER_URL_LIST, MMEMBED_SERVER_URL_LIST, MMQA_LLM_ANSWER_RESULT_FILE, MMQA_LLM_ANSWER_FAILED_FILE,
    MMQA_QUERY_CACHE_FILE,
    MMQA_FINAL_GRAPH_FILENAME, MMQA_GRAPH_RETRIEVAL_RESULT_FILENAME,
    BEAM_SIZE, TOP_K, MAX_HOP
)

def main():
    '''Component Embedder'''
    # sequential_component_embedder: SequentialComponentEmbedder = SequentialComponentEmbedder(
    #     MMQA_PARSE_JSON_FOLDER,
    #     MMQA_LDOC_FOLDER,
    #     MMQA_IMAGE_DESCRIPTION_INFO_FILE,
    #     MMQA_OBJECT_DETECT_INFO_FILE,
    #     MMQA_PROCESS_IMAGE_FOLDER
    # )
    # sequential_component_embedder.load_json_filelist()
    # sequential_component_embedder.run_embedding()
    
    '''Component MMQA Labeling'''
    # lilac_data_mmqa_mapper = LILaCDocMMQALabeler(
    #     MMQA_PATH,
    #     MMQA_LDOC_FOLDER,
    #     MMQA_IMAGE_REFERENCE_EMBEDDING_FOR_LABELING_FILE,
    #     MMQA_IMAGE_EMBEDDING_FOR_LABELING_FILE
    # )
    # lilac_data_mmqa_mapper.load_mmqa_reference()
    # lilac_data_mmqa_mapper.load_ldoc_from_folder()
    # lilac_data_mmqa_mapper.run_labeling(MMQA_LABELED_LDOC_FOLDER)
    # print(sum([len(idmap["txtid"]) + len(idmap["tabid"]) + len(idmap["imgid"]) for idmap in lilac_data_mmqa_mapper.doc_title_id_map.values()]))
    
    '''Query Caching'''
    # mmqa_cache_query_process(
    #     MMQA_PATH,
    #     QWEN_SERVER_URL_LIST[0],
    #     MMEMBED_SERVER_URL_LIST[0],
    #     "mmqa_query_caching_error_log.txt",
    #     MMQA_QUERY_CACHE_FILE
    # )
    
    '''Load Cached Data & Ground Truth Data'''
    # mmqa_cached_query_embedding_list: tp.List[MMQAQueryEmbedding] = mmqa_load_cached_query_data(MMQA_QUERY_CACHE_FILE)
    query_answer_list: tp.List[MMQAQueryAnswer] = mmqa_load_query_answer(MMQA_PATH)
    
    '''Verify MM-Embed Embedding'''
    # mmqa_embed_knn_test(query_answer_list, mmqa_cached_query_embedding_list, MMQA_LABELED_LDOC_FOLDER, TOP_K)
    
    '''Graph Construction'''
    # LILaCGraph.make_graph(MMQA_LABELED_LDOC_FOLDER, MMQA_FINAL_GRAPH_FILENAME)
    # lilac_graph: LILaCGraph = LILaCGraph.load_graph(MMQA_FINAL_GRAPH_FILENAME)
    
    '''LILaC Graph Retrieval'''
    # retrieval_result_list: tp.List[tp.Dict] = process_query_list_with_cached_data(
    #     MMQA_FINAL_GRAPH_FILENAME,
    #     mmqa_cached_query_embedding_list,
    #     BEAM_SIZE,
    #     TOP_K,
    #     MAX_HOP,
    #     MMQA_GRAPH_RETRIEVAL_RESULT_FILENAME
    # )
    # with open(MMQA_GRAPH_RETRIEVAL_RESULT_FILENAME, 'r', encoding='utf-8') as retrieval_result_file:
    #     retrieval_result_list: tp.List[tp.Dict] = [json.loads(result_file_line) for result_file_line in retrieval_result_file.readlines()]
    # mmqa_graph_retrieval_test(query_answer_list, retrieval_result_list)
    # mmqa_graph_retrieval_mrr_test(query_answer_list, retrieval_result_list)
    
    '''LLM Query'''
    # retrieval_result_map: tp.Dict[str, tp.List[tp.Dict]] = load_retrieval_result_map(lilac_graph.component_doc_title_map, MMQA_GRAPH_RETRIEVAL_RESULT_FILENAME)
    # query_retrieval_list: tp.List[tp.Tuple[MMQAQueryAnswer, tp.List[tp.Dict]]] = [
    #     (query_answer, retrieval_result_map[query_answer.qid])
    #     for query_answer in query_answer_list
    # ]
    # multiprocess_llm_answer(
    #     query_retrieval_list,
    #     QWEN_SERVER_URL_LIST[:3],
    #     MMQA_LLM_ANSWER_RESULT_FILE,
    #     MMQA_LLM_ANSWER_FAILED_FILE,
    #     MMQA_PROCESS_IMAGE_FOLDER
    # )
    
    '''Query Evaluation (Exact Match & F1 Score)'''
    llm_answer_list_map: tp.Dict[str, tp.List[str]] = dict()
    with open(MMQA_LLM_ANSWER_RESULT_FILE, "r", encoding="utf-8") as llm_answer_result_file:
        for llm_answer_line in llm_answer_result_file:
            json_line = json.loads(llm_answer_line)
            llm_answer_list_map[json_line["qid"]] = extract_answer_list_from_f_call(json_line["answer"])
    ground_truth_list_map: tp.Dict[str, tp.List[str]] = {
        query_answer.qid: query_answer.answer
        for query_answer in query_answer_list
    }
    mmqa_query_eval(llm_answer_list_map, ground_truth_list_map)

if __name__ == "__main__":
    main()
