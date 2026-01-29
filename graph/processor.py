import os
import re
import json
import typing as tp

from utils import *
import pysbd

class ProcessedComponent:
    def __init__(self) -> None:
        self.id: int = "" # unique id (filename + comp id)
        self.component_data = None # component json
        self.heading_path = [] # list(str)
        
        self.subcomp_embedding = [] # list(subcomp embed vector)
        self.edge = [] # list(comp id)

class LILaCDocument:
    def __init__(self, json_filepath: str, text_segmenter, img_folder: str) -> None:
        self.json_filepath = json_filepath
        self.img_folder = img_folder
        self.text_segmenter = text_segmenter

        self.doc_title = ""
        self.json_data = None
        
        self.processed_components: tp.List[ProcessedComponent] = []

    def load(self) -> bool:
        if not os.path.exists(self.json_filepath):
            print(f"Error: {self.json_filepath} not exists.")
            return False

        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as json_file:
                self.json_data = json.load(json_file)
            return True
        except json.JSONDecodeError:
            print("Error: incorrect JSON file format")
        except Exception as e:
            print(f"Error: {e}")
        
        return False
    
    def run(self) -> bool:
        # parsing
        self.doc_title = self.json_data["0"]["title"]
        
        for comp_key in self.json_data:
            if comp_key == "0":
                continue
            
            comp_id = f"{self.doc_title}_{comp_key}"
            comp_data = self.json_data[comp_key]
            result_comp = None
            if comp_data['type'] == "paragraph":
                continue # TODO
                result_comp = self.process_text_component(comp_data)
            elif comp_data['type'] == "table":
                result_comp = self.process_table_component(comp_data)
                break # TODO
            elif comp_data['type'] == "image":
                continue # TODO
                result_comp = self.process_image_component(comp_data)
            
            if result_comp is not None:
                result_comp.id = comp_id
                self.processed_components.append(result_comp)
        
        # document embedding
        # making subcomp
        # edge remapping
        
        return False
    
    def modality_check(self, sentence: str) -> str: # TODO
        return "Instruction: Represent the text for retrieval."
    
    def process_text_component(self, component) -> ProcessedComponent:
        result_comp = ProcessedComponent()
        sentences = self.text_segmenter.segment(component['paragraph'])
        subcomp_embedding = []
        for sentence in sentences:
            instruction = self.modality_check(sentence)
            subcomp_embedding.append(get_embedding(instruction, sentence, []))
        
        result_comp.component_data = component
        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"] # temporary
        result_comp.subcomp_embedding = subcomp_embedding
        
        return result_comp
    
    def flatten_table(self, table_data):
        pattern = r"\[\[([^\]]+)\]\]"
        image_path_list = []
        result_text_list = []

        for table_row in table_data:
            temp_list = []
            for table_elem in table_row:
                elem_img_lists = [self.get_clean_imagepath(item) for item in re.findall(pattern, table_elem)]
                image_path_list.extend(elem_img_lists)
                temp_list.append(re.sub(pattern, '', table_elem).strip())
            result_text_list.append(" [SEP] ".join(temp_list))
        result_text = " \n ".join(result_text_list)
        return result_text, image_path_list
    
    def process_table_component(self, component) -> ProcessedComponent:
        result_comp = ProcessedComponent()
        subcomp_embedding = []
        instruction = "Instruction: Represent the text for retrieval."
        original_table = component["table"]
        
        first_line = original_table[0]
        row_len = len(first_line)
        is_nm = all(isinstance(row, list) and len(row) == row_len for row in original_table)
        
        if is_nm: # NxM structure
            for table_line in original_table[1:]:
                text, img_paths = self.flatten_table([first_line, table_line])
                subcomp_embedding.append(get_embedding(instruction, text, img_paths))
        else: # not NxM structure (ex: infobox)
            for table_line in original_table:
                text, img_paths = self.flatten_table([table_line])
                print(instruction, text, img_paths)
                subcomp_embedding.append(get_embedding(instruction, text, img_paths))
        
        result_comp.component_data = component
        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"] # temporary
        
        full_text, full_img = self.flatten_table(component["table"])
        result_comp.subcomp_embedding = subcomp_embedding
                
        return result_comp
    
    def get_clean_imagepath(self, image_str):
        invalid_chars = '<>:"/\\|?*'
        
        clean_name = image_str
        if "File:" in clean_name:
            clean_name = clean_name.split("File:")[1]
        elif "https://" in clean_name:
            clean_name = clean_name.split("/")[-1]
        for char in invalid_chars:
            clean_name = clean_name.replace(char, '')

        full_path = os.path.join(self.img_folder, clean_name)
        return full_path
    
    def process_image_component(self, component) -> ProcessedComponent:
        result_comp = ProcessedComponent()
        instruction = "Express this images."
        
        full_path = self.get_clean_imagepath(self.img_folder, component["src"])

        result_comp.component_data = component
        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"]
        
        result_comp.subcomp_embedding = [get_embedding(instruction, component["caption"], [full_path])]

        return result_comp

class BatchDataProcessor:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path
        self.json_path_list = []
        self.lilac_doc_list = []
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def load(self) -> bool:
        if not os.path.exists(self.folder_path):
            return False

        self.json_path_list = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.folder_path, filename)
                self.json_path_list.append(file_path)

        return True

    def batch_run(self) -> bool:
        self.lilac_doc_list = []
        for json_path in self.json_path_list:
            new_doc = LILaCDocument(json_path)
            new_doc.load()
            new_doc.run(self.segmenter)
            self.lilac_doc_list.append(new_doc)
        return True

if __name__ == "__main__":
    # batch_data_processor = BatchDataProcessor('.')
    IMG_FOLDER = "/app/dataset/crawl/mmqa_image/"
    test_segmenter = pysbd.Segmenter(language="en", clean=False,)
    
    lilac_doc = LILaCDocument('test.json', test_segmenter, IMG_FOLDER)
    lilac_doc.load()
    lilac_doc.run()
