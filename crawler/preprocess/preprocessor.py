import pysbd

text = "Taylor Swift was born in West Reading, Penn. She later moved to Nashville, Tenn. at age 14."
segmenter = pysbd.Segmenter(language="en", clean=False)

sentences = segmenter.segment(text)
for i, sent in enumerate(sentences):
    print(f"[{i}] {sent}")

# [0] Taylor Swift was born in West Reading, Penn.
# [1] She later moved to Nashville, Tenn. at age 14.

class ProcessedComponent:
    def __init__(self, component_data) -> None:
        self.component_data = component_data
        self.embed = None # embed vector
        self.heading_path = []
        self.subcomponent = [] # (subcomp, embed) list
        self.edge = [] # list(str)

class Preprocessor:
    def __init__(self, json_filepath: str) -> None:
        self.json_filepath = json_filepath
        self.component = 
        self.processed = []

    def load(self) -> bool:
        return False

    def preprocess(self) -> bool:
        return False
    
    def embedding(self, data) -> bool:
        return False
    
    def make_subcomponent(self, data) -> bool:
        return False

class BatchPreprocessor:
    def __init__(self) -> None:
        pass
