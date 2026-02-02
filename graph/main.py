import sys
import time
import json

from lgraph import *
from processor import *
from query import *
from eval.mmqa import *
from utils import FINAL_RESULT_FILENAME, GRAPH_TEMP_FILE, LLM_TEMP_FILE

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
            tracker.elapsed_time = 0.0
            
            print(f"\nProcessing the query: {query}")
            with code_timer(tracker, "+ Get embedding... ", "Complete"):
                embedding = get_embedding(EmbeddingRequestData(query))
            with code_timer(tracker, "+ Get subembeddings... ", "Complete"):
                subembeddings = get_subembeddings(query)
            
            beam = LILaCBeam(lilac_graph, embedding, subembeddings, BEAM_SIZE)
            
            with code_timer(tracker, "+ Finding entry items from the graph... ", "Complete"):
                beam.find_entry()
            # print("Current beam:", [lilac_graph.comp_map[item] for item in beam.beam])
            
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

            print(f"Final Components: {final_comp_ids}. Total elapsed time: {tracker.elapsed_time:.3f}s")
            
            temp_json_file.write(json.dumps({'query': query, 'comp_ids': final_comp_ids}, ensure_ascii=False) + '\n')
            temp_json_file.flush()

            result_comps_list.append(beam.top_comps(top_k=TOP_K))
            result_docs_list.append(beam.top_doc_titles(top_k=TOP_K))
            elapsed_time_list.append(tracker.elapsed_time)
    
    return result_comps_list, result_docs_list, elapsed_time_list

def main(query_list: list[str], temp_graph_filepath: str, temp_llm_filepath: str):
    query_list = sorted(query_list)
    result_comps_list, result_docs_list, elapsed_time_list = process_query_list(query_list, temp_graph_filepath)
    print("\n", "+ Total Result\n")
    print(f"Mean Elapsed Time: {sum(elapsed_time_list) / len(elapsed_time_list):.3f}s")
    print(f"Total Elapsed Time: {sum(elapsed_time_list):.3f}s\n\n")
    llm_response_list = []
    with open(temp_llm_filepath, "a", encoding="utf-8") as f_out:
        for ind in range(len(query_list)):
            q = query_list[ind]
            print(f"[{ind+1}/{len(query_list)}] Processing query: {q[:50]}...")
            
            final_query, img_paths = llm_question_query(q, IMG_FOLDER, result_docs_list[ind], result_comps_list[ind])
            
            try:
                llm_response = get_llm_response("", final_query, img_paths)
            except RuntimeError as e:
                print(f"!!! Error at query {ind}: {e}")
                llm_response = "ERROR: Generation Failed"

            llm_response_list.append(llm_response)
            print(f"=> LLM Augmented Answer: {llm_response}\n")
            
            f_out.write(json.dumps({"query": q, "answer": llm_response}, ensure_ascii=False) + "\n")
            f_out.flush()
        
    return llm_response_list, result_comps_list

if __name__ == "__main__":
    query_answer_list = mmqa_eval_load(
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
