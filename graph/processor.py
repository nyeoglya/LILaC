import os
import re
import json
import pickle
import typing as tp

import numpy as np

from utils import *
import pysbd

class ProcessedComponent:
    def __init__(self, component) -> None:
        self.id: int = 0 # unique id
        self.file_name: str = ""
        self.component = component
        self.heading_path = [] # list(str)
        
        self.embedding: np.array = np.array([])
        self.subcomp_embeddings: tp.List[np.array] = [] # list(subcomp embed vector id)
        self.edge: tp.List[int] = [] # list(comp unique id)

class LILaCDocument:
    def __init__(self, json_filepath: str, text_segmenter, img_folder: str) -> None:
        self.json_filepath = json_filepath
        self.img_folder = img_folder
        self.text_segmenter = text_segmenter

        self.doc_title = ""
        self.json_data = None
        
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
        
        comp_id = start_comp_id
        for comp_key in self.json_data:
            if comp_key == "0":
                continue
            
            comp_data = self.json_data[comp_key]
            result_comp = None
            try:
                if comp_data['type'] == "paragraph":
                    result_comp = self.process_text_component(comp_data)
                elif comp_data['type'] == "table":
                    result_comp = self.process_table_component(comp_data)
                elif comp_data['type'] == "image":
                    result_comp = self.process_image_component(comp_data)
            except:
                print(f"ERROR: {comp_data}")
                exit(-1)
            
            if result_comp is not None:
                result_comp.id = comp_id
                result_comp.file_name = self.doc_title
                self.processed_components.append(result_comp)
            comp_id += 1
        
        return comp_id
    
    def process_text_component(self, component) -> ProcessedComponent:
        result_comp = ProcessedComponent(component)
        sentences = self.text_segmenter.segment(component['paragraph'])
        subcomp_requests = []
        for sentence in sentences:
            subcomp_requests.append(EmbeddingRequestData(sentence, ""))
        
        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"] # temporary
        
        result_comp.embedding = get_embedding(EmbeddingRequestData(component['paragraph'].replace("\n",""), ""))
        result_comp.subcomp_embeddings = get_batch_embedding(subcomp_requests)
        return result_comp
    
    def flatten_table(self, table_data):
        pattern = r"\[\[([^\]]+)\]\]"
        image_path_list = []
        result_text_list = []

        for table_row in table_data:
            temp_list = []
            for table_elem in table_row:
                elem_img_lists = [get_clean_imagepath(self.img_folder, item) for item in re.findall(pattern, table_elem)]
                image_path_list.extend(elem_img_lists)
                text = re.sub(pattern, '', table_elem).strip()
                if text:
                    temp_list.append(text)
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
        result_comp = ProcessedComponent(component)
        subcomp_embeddings = []
        original_table = component["table"]
        
        first_line = original_table[0]
        row_len = len(first_line)
        is_nm = all(isinstance(row, list) and len(row) == row_len for row in original_table)
        
        if is_nm: # NxM structure
            if len(original_table) == 1:
                text, img_paths = self.flatten_table([first_line])
                subcomp_embeddings.append(get_embedding(EmbeddingRequestData(text, img_paths[0] if img_paths else "")))
            else:
                for table_line in original_table[1:]:
                    text, img_paths = self.flatten_table([first_line, table_line])
                    subcomp_embeddings.append(get_embedding(EmbeddingRequestData(text, img_paths[0] if img_paths else "")))
        else: # not NxM structure (ex: infobox)
            for table_line in original_table:
                text, img_paths = self.flatten_table([table_line])
                subcomp_embeddings.append(get_embedding(EmbeddingRequestData(text, img_paths[0] if img_paths else "")))
        
        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"] # temporary
        
        result_comp.subcomp_embeddings = subcomp_embeddings
        result_comp.embedding = np.mean(np.stack(subcomp_embeddings, axis=0), axis=0)
        
        return result_comp
        
    def process_image_component(self, component) -> ProcessedComponent:
        result_comp = ProcessedComponent(component)
        
        full_path = get_clean_imagepath(self.img_folder, component["src"])
        
        if not os.path.exists(full_path):
            print(f"Error: No {full_path} exists")
            return None

        result_comp.heading_path = component["heading_path"]
        result_comp.edge = component["edge"]
        
        result_comp.embedding = get_embedding(EmbeddingRequestData(component["caption"], full_path))
        result_comp.subcomp_embeddings = [result_comp.embedding]

        return result_comp

class BatchDataProcessor:
    def __init__(self, json_folder_path: str, img_folder: str, ldoc_folder_path: str) -> None:
        self.json_folder_path: str = json_folder_path
        self.img_folder: str = img_folder
        self.ldoc_folder_path: str = ldoc_folder_path
        
        self.json_path_list: tp.List[str] = []
        self.lilac_doc_dict: tp.Dict[str, LILaCDocument] = dict()
        
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def load(self) -> bool:
        if not os.path.exists(self.ldoc_folder_path):
            return False

        self.lilac_doc_dict = {}
        for filename in os.listdir(self.ldoc_folder_path):
            if filename.endswith(".ldoc"):
                file_path = os.path.join(self.ldoc_folder_path, filename)
                new_doc = LILaCDocument.load(file_path)
                self.lilac_doc_dict[new_doc.doc_title] = new_doc

        return True

    def load_json(self) -> bool:
        if not os.path.exists(self.json_folder_path):
            return False

        self.json_path_list = []
        for filename in os.listdir(self.json_folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.json_folder_path, filename)
                self.json_path_list.append(file_path)

        return True

    def edge_remapping(self) -> bool:
        if not len(self.lilac_doc_dict):
            return False

        edge_range_map: tp.Dict[str, tp.Tuple[int, int]] = dict()
        for doc_title in self.lilac_doc_dict:
            lilac_doc = self.lilac_doc_dict[doc_title]
            if not lilac_doc.processed_components:
                continue
            first_comp_id = lilac_doc.processed_components[0].id
            last_comp_id = lilac_doc.processed_components[-1].id
            edge_range_map[doc_title] = list(range(first_comp_id, last_comp_id + 1))
        
        for doc_title in self.lilac_doc_dict:
            lilac_doc = self.lilac_doc_dict[doc_title]
            for processed_comp in lilac_doc.processed_components:
                id_edge_list = []
                for edge_name in processed_comp.edge:
                    edge_range = edge_range_map.get(edge_name)
                    if edge_range:
                        id_edge_list.extend(edge_range)
                processed_comp.edge = id_edge_list

        for remapped_doc in self.lilac_doc_dict.values():
            remapped_doc.save(os.path.join(self.ldoc_folder_path, f"{remapped_doc.doc_title}.ldoc.remapped"))

        return True

    def batch_run(self) -> bool:
        comp_id = 0
        for json_path in self.json_path_list:
            new_doc = LILaCDocument(json_path, self.segmenter, self.img_folder)
            new_doc.load_json()
            new_doc_title = new_doc.json_data["0"]["title"]
            new_ldoc_path = os.path.join(self.ldoc_folder_path, f"{new_doc_title}.ldoc")
            if os.path.exists(new_ldoc_path):
                print(f"Skip document {new_doc_title} as it is already parsed.")
                continue
            
            try:
                comp_id = new_doc.run(comp_id)
                new_doc.save(new_ldoc_path)
                self.lilac_doc_dict[new_doc.doc_title] = new_doc
            except:
                print(f"Skip document {new_doc_title} as it failed")
        return True

if __name__ == "__main__":
    batch_data_processor = BatchDataProcessor(JSON_FOLDER, IMG_FOLDER, LDOC_FOLDER)
    test_segmenter = pysbd.Segmenter(language="en", clean=False,)
    batch_data_processor.load_json()
    batch_data_processor.batch_run()
    batch_data_processor.edge_remapping()
