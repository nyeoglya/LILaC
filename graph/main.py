import typing as tp

from embed import SequentialComponentEmbedder
from graph import LILaCGraph, LILaCDocument, ProcessedComponent
from utils.mmqa import (
    MMQAQueryEmbedding,
    mmqa_load_query_answer, mmqa_cache_query_process, mmqa_load_cached_query_data
)
from test import mmqa_embed_test
from remap import LILaCDocMMQAMapper

from config import (
    MMQA_PATH,
    MMQA_PARSE_JSON_FOLDER,
    MMQA_LDOC_FOLDER,
    MMQA_IMAGE_DESCRIPTION_INFO_FILE,
    MMQA_OBJECT_DETECT_INFO_FILE,
    MMQA_PROCESS_IMAGE_FOLDER,
    MMQA_REMAP_IMAGE_EMBEDDING_FILE,
    MMQA_REMAP_IMAGE_REFERENCE_EMBEDDING_FILE,
    MMQA_REMAPPED_LDOC_FOLDER,
    QWEN_SERVER_URL_LIST,
    MMEMBED_SERVER_URL_LIST,
    MMQA_QUERY_CACHE_FILE,
    MMQA_FINAL_GRAPH_FILENAME,
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
    
    '''Verify MM-Embed Embedding'''
    mmqa_cached_query_data_list: tp.List[MMQAQueryEmbedding] = mmqa_load_cached_query_data(MMQA_QUERY_CACHE_FILE)
    query_answer_list = mmqa_load_query_answer(MMQA_PATH)
    mmqa_embed_test(query_answer_list, mmqa_cached_query_data_list, MMQA_REMAPPED_LDOC_FOLDER)
    
    '''Graph Construction'''
    # LILaCGraph.make_graph(MMQA_REMAPPED_LDOC_FOLDER, MMQA_FINAL_GRAPH_FILENAME)
    # lilac_graph: LILaCGraph = LILaCGraph.load_graph(MMQA_FINAL_GRAPH_FILENAME)
    
    '''LILaC Query''' # TODO
    # llm_response, result_comps_list = main(query_list, GRAPH_TEMP_FILE, LLM_TEMP_FILE)
    # for ind in range(len(query_answer_list)):
    #     query_answer_list[ind].llm_answer = llm_response[ind]
    #     query_answer_list[ind].result_comps = result_comps_list[ind]
    
    # users_list_of_dict = [asdict(user) for user in query_answer_list]
    # with open(FINAL_RESULT_FILENAME, "w", encoding="utf-8") as f:
    #     json.dump(users_list_of_dict, f, indent=4, ensure_ascii=False)
    # print(f"+ Save result to {FINAL_RESULT_FILENAME}")
    
    # from query import subquery_divide_query
    # sub_query_list = [get_llm_response(QWEN_SERVER_URL_LIST[0], subquery_divide_query(text)).replace("\n","").split(";") for text in query_list]
    # for i in range(len(query_answer_list)):
    #     print(f"Q: {query_list[i]}")
    #     print(f"Subqueries: {sub_query_list[i]}") # 분해된 결과 확인
    #     # print(f"Retrieved IDs: {result_comps_list[i]}") # 찾은 컴포넌트 ID
    #     print(f"Actual Answer: {query_answer_list[i].answer}") # 정답지
    #     print(f"LLM Answer: {llm_response[i]}") # 모델이 내뱉은 말
    #     print("-" * 30)
    
    '''LILaC Evaluation''' # TODO
    # mmqa_query_eval(query_answer_list)

if __name__ == "__main__":
    main()
