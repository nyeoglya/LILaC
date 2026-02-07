import pickle
import typing as tp

from embed import SequentialComponentEmbedder
from graph import LILaCGraph, LILaCDocument, ProcessedComponent
from utils.mmqa import (
    MMQAQueryEmbedding,
    mmqa_load_query_answer, mmqa_cache_query_process, mmqa_load_cached_query_data
)
from test import mmqa_embed_knn_test, mmqa_graph_retrieval_test
from remap import LILaCDocMMQAMapper
from process import process_query_list_with_cached_data
from config import (
    MMQA_PATH, MMQA_LDOC_FOLDER, MMQA_PROCESS_IMAGE_FOLDER,
    MMQA_PARSE_JSON_FOLDER,
    MMQA_IMAGE_DESCRIPTION_INFO_FILE, MMQA_OBJECT_DETECT_INFO_FILE,
    MMQA_REMAP_IMAGE_EMBEDDING_FILE, MMQA_REMAP_IMAGE_REFERENCE_EMBEDDING_FILE, MMQA_REMAPPED_LDOC_FOLDER,
    QWEN_SERVER_URL_LIST, MMEMBED_SERVER_URL_LIST,
    MMQA_QUERY_CACHE_FILE,
    MMQA_FINAL_GRAPH_FILENAME, MMQA_FINAL_QUERY_ANSWER_FILENAME, MMQA_GRAPH_RETRIEVAL_RESULT_FILENAME,
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
    
    '''Component MMQA Mapper'''
    # lilac_data_mmqa_mapper = LILaCDocMMQAMapper(
    #     MMQA_PATH,
    #     MMQA_LDOC_FOLDER,
    #     MMQA_REMAP_IMAGE_REFERENCE_EMBEDDING_FILE,
    #     MMQA_REMAP_IMAGE_EMBEDDING_FILE
    # )
    # lilac_data_mmqa_mapper.load_mmqa_reference()
    # lilac_data_mmqa_mapper.load_ldoc_from_folder()
    # lilac_data_mmqa_mapper.run_remapping(MMQA_REMAPPED_LDOC_FOLDER)
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
    mmqa_cached_query_embedding_list: tp.List[MMQAQueryEmbedding] = mmqa_load_cached_query_data(MMQA_QUERY_CACHE_FILE)
    query_answer_list = mmqa_load_query_answer(MMQA_PATH)
    
    '''Verify MM-Embed Embedding'''
    mmqa_embed_knn_test(query_answer_list, mmqa_cached_query_embedding_list, MMQA_REMAPPED_LDOC_FOLDER, 9)
    
    '''Graph Construction'''
    # LILaCGraph.make_graph(MMQA_REMAPPED_LDOC_FOLDER, MMQA_FINAL_GRAPH_FILENAME)
    # lilac_graph: LILaCGraph = LILaCGraph.load_graph(MMQA_FINAL_GRAPH_FILENAME)
    
    '''LILaC Graph Retrieval'''
    retrieval_result_list: tp.List[tp.Dict] = process_query_list_with_cached_data(
        MMQA_FINAL_GRAPH_FILENAME,
        mmqa_cached_query_embedding_list,
        BEAM_SIZE,
        TOP_K,
        MAX_HOP,
        MMQA_GRAPH_RETRIEVAL_RESULT_FILENAME
    )
    mmqa_graph_retrieval_test(query_answer_list, retrieval_result_list)
    
    '''LLM Query''' # TODO
    # llm_response_list = batch_llm_response(result_components_list)
    # for ind in range(len(query_answer_list)):
    #     query_answer_list[ind].llm_answer = llm_response_list[ind]
    #     query_answer_list[ind].result_comps = result_components_list[ind][""]
    
    # with open(???, "w", encoding="utf-8") as query_answer_list_file:
    #     pickle.dump(query_answer_list)
    # print(f"+ Save result to {MMQA_FINAL_QUERY_ANSWER_FILENAME}")
    
    '''Query Evaluation''' # TODO
    # mmqa_query_eval(query_answer_list)

if __name__ == "__main__":
    main()
