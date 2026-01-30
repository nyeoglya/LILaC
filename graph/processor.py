import os
import re
import json
import typing as tp

from utils import *
import pysbd

import pickle

class ProcessedComponent:
    def __init__(self) -> None:
        self.id: int = 0 # unique id
        self.file_name: str = ""
        self.heading_path = [] # list(str)
        
        self.subcomp_embedding = [] # list(subcomp embed vector id)
        self.edge: tp.List[int] = [] # list(comp unique id)

class LILaCDocument:
    def __init__(self, json_filepath: str, text_segmenter, img_folder: str) -> None:
        self.json_filepath = json_filepath
        self.img_folder = img_folder
        self.text_segmenter = text_segmenter

        self.doc_title = ""
        self.json_data = None
        
        self.embedding = None
        self.processed_components: tp.List[ProcessedComponent] = []

    def save(self, save_path: str) -> bool:
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Successfully saved to {save_path}")
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False

    @staticmethod
    def load(load_path: str):
        try:
            if not os.path.exists(load_path):
                print("File not found.")
                return None
            
            with open(load_path, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except Exception as e:
            print(f"Load failed: {e}")
            return None

    def load_json(self) -> bool:
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
    
    def run(self, start_comp_id) -> int:
        # parsing
        self.doc_title = self.json_data["0"]["title"]
        
        text_list = []
        comp_id = start_comp_id
        for comp_key in self.json_data:
            if comp_key == "0":
                continue
            
            comp_data = self.json_data[comp_key]
            result_comp = None
            if comp_data['type'] == "paragraph":
                result_comp, text = self.process_text_component(comp_data)
                text_list.append(text)
            elif comp_data['type'] == "table":
                result_comp, text = self.process_table_component(comp_data)
                text_list.append(text)
            elif comp_data['type'] == "image":
                result_comp = self.process_image_component(comp_data)
            
            if result_comp is not None:
                result_comp.id = comp_id
                result_comp.file_name = self.doc_title
                self.processed_components.append(result_comp)
            comp_id += 1
        
        # document embedding
        full_text = "".join(text_list)
        self.embedding = get_embedding(EmbeddingRequestData("Express this document.", full_text, ""))
        
        return comp_id
    
    def process_text_component(self, component) -> ProcessedComponent:
        result_comp = ProcessedComponent()
        sentences = self.text_segmenter.segment(component['paragraph'])
        subcomp_requests = []
        instruction = "Express this table."
        for sentence in sentences:
            subcomp_requests.append(EmbeddingRequestData(instruction, sentence, ""))
        
        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"] # temporary
        result_comp.subcomp_embedding = get_batch_embedding(subcomp_requests)
        
        return result_comp, component['paragraph']
    
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
        
        clean_image_path_list = []
        for image_path in image_path_list:
            if os.path.exists(image_path):
                clean_image_path_list.append(image_path)
            else:
                print(f"Error: no {image_path} exists")
        
        return result_text, clean_image_path_list
    
    def process_table_component(self, component) -> ProcessedComponent:
        result_comp = ProcessedComponent()
        subcomp_requests = []
        instruction = "Express this table."
        original_table = component["table"]
        
        first_line = original_table[0]
        row_len = len(first_line)
        is_nm = all(isinstance(row, list) and len(row) == row_len for row in original_table)
        
        if is_nm: # NxM structure
            if len(original_table) == 1:
                text, img_paths = self.flatten_table([first_line])
                subcomp_requests.append(EmbeddingRequestData(instruction, text, img_paths[0] if img_paths else ""))
            else:
                for table_line in original_table[1:]:
                    text, img_paths = self.flatten_table([first_line, table_line])
                    subcomp_requests.append(EmbeddingRequestData(instruction, text, img_paths[0] if img_paths else ""))
        else: # not NxM structure (ex: infobox)
            for table_line in original_table:
                text, img_paths = self.flatten_table([table_line])
                subcomp_requests.append(EmbeddingRequestData(instruction, text, img_paths[0] if img_paths else ""))
        
        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"] # temporary
        
        full_text, _ = self.flatten_table(original_table)
        
        result_comp.subcomp_embedding = get_batch_embedding(subcomp_requests)
        
        return result_comp, full_text
    
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
        
        full_path = self.get_clean_imagepath(component["src"])
        
        if not os.path.exists(full_path):
            print(f"Error: No {full_path} exists")
            return None

        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"]
        
        result_comp.subcomp_embedding = [get_embedding(EmbeddingRequestData(instruction, component["caption"], full_path))]

        return result_comp

class BatchDataProcessor:
    def __init__(self, folder_path, img_folder, save_folder) -> None:
        self.folder_path = folder_path
        self.img_folder = img_folder
        self.save_folder = save_folder
        
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

    def edge_remapping(self):
        pass

    def batch_run(self) -> bool:
        self.lilac_doc_list = []
        comp_id = 0
        for json_path in self.json_path_list:
            new_doc = LILaCDocument(json_path, self.segmenter, self.img_folder)
            new_doc.load_json()
            comp_id = new_doc.run(comp_id)
            new_doc.save(os.path.join(self.save_folder, f"{new_doc.doc_title}.ldoc"))
        return True

if __name__ == "__main__":
    JSON_FOLDER = "/dataset/crawl/mmqa_html_top5/"
    IMG_FOLDER = "/dataset/crawl/mmqa_image/"
    SAVE_FOLDER = "/dataset/process/mmqa/"
    
    batch_data_processor = BatchDataProcessor(JSON_FOLDER, IMG_FOLDER, SAVE_FOLDER)
    test_segmenter = pysbd.Segmenter(language="en", clean=False,)
    batch_data_processor.load()
    print(batch_data_processor.json_path_list)
    batch_data_processor.batch_run()
    
    '''
    lilac_doc = LILaCDocument('test.json', test_segmenter, IMG_FOLDER)
    lilac_doc.load_json()
    lilac_doc.run(0)
    lilac_doc.save('test.ldoc')
    '''
