import os
import sys
import time
import json

from lgraph import LILaCGraph, LILaCBeam
from query import get_subembeddings, llm_question_query
from eval.mmqa import mmqa_load
from eval.utils import query_eval, QueryAnswer
from utils import (
    get_embedding, get_llm_response,
    EmbeddingRequestData,
    IMG_FOLDER, FINAL_RESULT_FILENAME, GRAPH_TEMP_FILE, LLM_TEMP_FILE
)

from dataclasses import asdict
from contextlib import contextmanager

class TimeTracker:
    def __init__(self):
        self.elapsed_time = 0.0

@contextmanager
def code_timer(tracker, before, label):
    print(f"{before}", end="", flush=True)
    start = time.time()
    yield
    delta = time.time() - start
    if tracker:
        tracker.elapsed_time += delta
        print(f"{label}: {tracker.elapsed_time:.3f}s (+{delta:.3f}s)")
    else:
        print(f"{label} ({delta:.3f}s passed)")

# end-to-end pipeline (with given datasets)
def process_query_list(query_list: list[str], temp_filepath: str):
    BEAM_SIZE = 30
    TOP_K = 3
    MAX_HOP = 10
    GRAPH_FILE_PATH = "wiki.lgraph"
    
    # 1. 기존 결과 로드 (캐싱)
    cache = {}
    if os.path.exists(temp_filepath):
        with open(temp_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    cache[data['query']] = data
                except: continue
    
    # 그래프 로딩 타이머 유지
    tracker = TimeTracker()
    with code_timer(None, "+ Start loading LILaCGraph... ", "Complete"):
        lilac_graph = LILaCGraph(GRAPH_FILE_PATH)
        if not lilac_graph.load():
            sys.exit(-1)
    
    result_comps_list: tp.List[tp.List[dict]] = []
    result_docs_list: tp.List[tp.List[str]] = []
    elapsed_time_list: tp.List[float] = []

    with open(temp_filepath, 'a', encoding='utf-8') as temp_json_file:
        for query in query_list:
            # 캐시에 있는 경우: 타이머 없이 데이터만 복구
            if query in cache:
                cached_data = cache[query]
                result_comps_list.append(cached_data.get('comps', []))
                result_docs_list.append(cached_data.get('docs', []))
                elapsed_time_list.append(cached_data.get('elapsed_time', 0.0))
                print(f"Skipping (cached): {query[:50]}...")
                continue

            # 캐시에 없는 경우: 원래의 code_timer 로직 그대로 실행
            tracker.elapsed_time = 0.0
            print(f"\nProcessing the query: {query}")
            
            with code_timer(tracker, "+ Get embedding... ", "Complete"):
                embedding = get_embedding(EmbeddingRequestData(query))
            
            with code_timer(tracker, "+ Get subembeddings... ", "Complete"):
                subembeddings = get_subembeddings(query)
            
            beam = LILaCBeam(lilac_graph, embedding, subembeddings, BEAM_SIZE)
            
            with code_timer(tracker, "+ Finding entry items from the graph... ", "Complete"):
                beam.find_entry()
            
            hop_count = 0
            while hop_count < MAX_HOP:
                with code_timer(tracker, f"+ Trying one hop (step {hop_count+1})... ", "Complete"):
                    changed = beam.one_hop()
                
                if not changed:
                    print(f"+ Convergence reached at {hop_count+1} hops.")
                    break
                hop_count += 1

            with code_timer(tracker, "+ Finding the final components... ", "Complete"):
                final_comp_ids = beam.top_comp_ids(top_k=TOP_K)
                # 캐시 복구를 위해 실제 객체 데이터도 가져옴
                comps = beam.top_comps(top_k=TOP_K)
                docs = beam.top_doc_titles(top_k=TOP_K)

            print(f"Final Components: {final_comp_ids}. Total elapsed time: {tracker.elapsed_time:.3f}s")
            
            # 나중에 복구할 때 필요한 정보를 모두 저장
            record = {
                'query': query, 
                'comp_ids': final_comp_ids,
                'comps': comps,
                'docs': docs,
                'elapsed_time': tracker.elapsed_time
            }
            temp_json_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            temp_json_file.flush()

            result_comps_list.append(comps)
            result_docs_list.append(docs)
            elapsed_time_list.append(tracker.elapsed_time)
    
    return result_comps_list, result_docs_list, elapsed_time_list

def main(query_list: list[str], temp_graph_filepath: str, temp_llm_filepath: str):
    query_list = sorted(query_list)
    
    # 1. 그래프 검색 결과 (이미 process_query_list에서 캐싱 처리됨)
    result_comps_list, result_docs_list, elapsed_time_list = process_query_list(query_list, temp_graph_filepath)
    
    print("\n", "+ Total Result\n")
    # 0.0보다 큰(새로 계산된) 시간들만 평균 계산
    new_times = [t for t in elapsed_time_list if t > 0]
    if new_times:
        print(f"Mean Elapsed Time (New): {sum(new_times) / len(new_times):.3f}s")
    print(f"Total Elapsed Time (All): {sum(elapsed_time_list):.3f}s\n\n")

    # 기존 LLM 응답 로드 (중복 실행 방지)
    llm_cache = {}
    if os.path.exists(temp_llm_filepath):
        with open(temp_llm_filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    llm_cache[data['query']] = data['answer']
                except: continue
        print(f"이미 완료된 {len(llm_cache)}개의 LLM 응답을 로드했습니다.")

    llm_response_list = []
    
    # LLM 처리 루프
    with open(temp_llm_filepath, "a", encoding="utf-8") as f_out:
        for ind in range(len(query_list)):
            q = query_list[ind]
            
            # 이미 답변이 있는 경우 캐시에서 가져오고 건너뜀
            if q in llm_cache:
                print(f"[{ind+1}/{len(query_list)}] Skipping (already answered): {q[:30]}...")
                llm_response_list.append(llm_cache[q])
                continue

            # 새로운 답변 생성
            print(f"[{ind+1}/{len(query_list)}] Processing LLM query: {q[:50]}...")
            
            # process_query_list에서 반환한 전체 리스트의 인덱스 활용
            final_query, img_paths = llm_question_query(
                q, IMG_FOLDER, result_docs_list[ind], result_comps_list[ind]
            )
            
            try:
                # 이 부분에 별도의 code_timer가 필요하다면 감쌀 수 있습니다.
                llm_response = get_llm_response("", final_query, img_paths)
            except RuntimeError as e:
                print(f"!!! Error at query {ind}: {e}")
                llm_response = "ERROR: Generation Failed"

            # 결과 저장 및 기록
            llm_response_list.append(llm_response)
            print(f"=> LLM Augmented Answer: {llm_response[:100]}\n")
            
            f_out.write(json.dumps({"query": q, "answer": llm_response}, ensure_ascii=False) + "\n")
            f_out.flush()
        
    return llm_response_list, result_comps_list

if __name__ == "__main__":
    query_answer_list = mmqa_load(
        "/dataset/mmqa/MMQA_dev.jsonl",
        "/dataset/mmqa/MMQA_texts.jsonl",
        "/dataset/mmqa/MMQA_images.jsonl",
        "/dataset/mmqa/MMQA_tables.jsonl",
    )
    
    query_list = [query_answer.question for query_answer in query_answer_list]
    llm_response, result_comps_list = main(query_list, GRAPH_TEMP_FILE, LLM_TEMP_FILE)
    for ind in range(len(query_answer_list)):
        query_answer_list[ind].llm_answer = llm_response[ind]
        query_answer_list[ind].result_comps = result_comps_list[ind]
    
    users_list_of_dict = [asdict(user) for user in query_answer_list]
    with open(FINAL_RESULT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(users_list_of_dict, f, indent=4, ensure_ascii=False)
    print(f"+ Save result to {FINAL_RESULT_FILENAME}")
    
    
    
