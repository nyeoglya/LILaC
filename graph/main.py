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
def main():
    BEAM_SIZE = 5
    GRAPH_FILE_PATH = "wiki.lgraph"
    
    tracker = TimeTracker()
    
    with code_timer(tracker, "+ Get subembeddings... ", "Complete"):
        subembeddings = get_subembeddings("For which film did Ben Piazza play the role of Mr. Simms?")
    with code_timer(None, "+ Start loading LILaCGraph... ", "Complete"):
        lilac_graph = LILaCGraph(GRAPH_FILE_PATH)
        lilac_graph.load()
    
    with code_timer(tracker, "+ Finding entry items from the graph... ", "Complete"):
        beam = find_entry(lilac_graph, subembeddings, BEAM_SIZE)
    print("Current beam:", [lilac_graph.comp_map[item] for item in beam])
    
    with code_timer(tracker, "+ Trying one hop... ", "Complete"):
        beam = one_hop(lilac_graph, subembeddings, beam)
    print("Current beam:", [lilac_graph.comp_map[item] for item in beam])
    
    with code_timer(tracker, "+ Finding the final components... ", "Complete"):
        comp1, comp2 = final_edge(lilac_graph, subembeddings, beam)
    print(f"Final Component 1: {comp1}")
    # print(f"Final Component 2: {comp2}")


if __name__ == "__main__":
    main()
