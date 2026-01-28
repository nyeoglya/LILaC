import os
import json
import pysbd

from base import *

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

class DataProcessor:
    def __init__(self, json_filepath: str) -> None:
        self.json_filepath = json_filepath
        self.component: tp.Union[ComponentData, None] = None

    def load(self) -> bool:
        if not os.path.exists(self.json_filepath):
            print(f"Error: {self.json_filepath} not exists.")
            return False

        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as json_file:
                self.data = json.load(json_file)
            
            self.data[0]
            
            return True            
        except json.JSONDecodeError:
            print("Error: incorrect JSON file format")
        except Exception as e:
            print(f"Error: {e}")
            
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
