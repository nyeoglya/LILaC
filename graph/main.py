import sys
import time

from lgraph import *
from processor import *
from query import *

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
def main(query_list: list[str]):
    BEAM_SIZE = 3
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
    for query in query_list:
        print(f"\nProcessing the query: {query}")
        with code_timer(tracker, "+ Get subembeddings... ", "Complete"):
            subembeddings = get_subembeddings(query)
        
        beam = LILaCBeam(lilac_graph, subembeddings, BEAM_SIZE)
        
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
        
        result_comps_list.append(beam.top_comps(top_k=TOP_K))
        result_docs_list.append(beam.top_doc_titles(top_k=TOP_K))
        elapsed_time_list.append(tracker.elapsed_time)
    return result_comps_list, result_docs_list, elapsed_time_list

if __name__ == "__main__":
    query_list = [
        "Which film did Ben Piazza play the role of Mr. Simms?"
    ]
    result_comps_list, result_docs_list, elapsed_time_list = main(query_list)
    print("\n", "-"*20, "Total Result", "-"*20, "\n")
    print(f"Mean Elapsed Time: {sum(elapsed_time_list) / len(elapsed_time_list):.3f}s\n")
    for docs, query in zip(result_docs_list, query_list):
        print(f"Query: {query}\n=> Result Docs List: {docs}")
    
    for ind in range(len(query_list)):
        final_query, img_paths = llm_question_query(query_list[ind], IMG_FOLDER, result_docs_list[ind], result_comps_list[ind])
        print(final_query)
        llm_response = get_llm_response("", final_query, img_paths)
        print(llm_response)
    
    print("\n", result_comps_list)
