import os
import json
import typing as tp

import pysbd

class SubComponent:
    def __init__(self, element) -> None:
        self.embedding = None
        self.element = element

class ProcessedComponent:
    def __init__(self, sub_id: int, component_data) -> None:
        self.sub_id: int = sub_id
        self.component_data = component_data # TODO
        self.embedding = None # embed vector
        self.heading_path = []
        self.sub_component = []
        self.edge = [] # list(str)

class LILaCDocument:
    def __init__(self, json_filepath: str) -> None:
        self.json_filepath = json_filepath
        self.processed_component: tp.List[ProcessedComponent] = []

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
    
    def run(self) -> bool:
        # parsing & document embedding
        # making subcomponenet
        return False
    
    def make_subcomponent(self, data) -> bool:
        return False
    
    def embedding(self, data) -> bool:
        return False


class BatchDataProcessor:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path

    def batch_run(self) -> bool:
        return False


if __name__ == "__main__":    
    text = "Taylor Swift was born in West Reading, Penn. She later moved to Nashville, Tenn. at age 14."
    segmenter = pysbd.Segmenter(language="en", clean=False)

    sentences = segmenter.segment(text)
    for i, sent in enumerate(sentences):
        print(f"[{i}] {sent}")
