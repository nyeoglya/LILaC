import os
import re
import json
import pickle
import typing as tp

from tqdm import tqdm
import numpy as np

from utils import (
    get_embedding, get_batch_embedding, get_clean_savepath_from_url,
    EmbeddingRequestData,
    PARSED_JSON_FOLDER, IMG_FOLDER, LDOC_FOLDER
)
import pysbd

class ProcessedComponent:
    def __init__(self, original_component) -> None:
        self.component_uuid: str = ""
        self.original_json_filepath: str = ""
        self.original_component = original_component
        self.component_embedding: np.ndarray = np.array([])
        self.subcomponent_embeddings: tp.List[np.ndarray] = [] # list(subcomp embed vector)
        self.neighbor_components: tp.List[str] = [] # list(comp unique id)

class LILaCDocument:
    def __init__(
        self,
        json_filepath: str,
        text_segmenter,
        img_folder: str
    ) -> None:
        self.json_filepath: str = json_filepath
        self.img_folder: str = img_folder
        self.text_segmenter = text_segmenter

        self.doc_title: str = ""
        self.json_data: tp.Dict = dict()
        
        self.processed_components: tp.List[ProcessedComponent] = []

    def save(self, save_path: str) -> bool:
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
            tqdm.write(f"Successfully saved to {save_path}")
            return True
        except Exception as e:
            tqdm.write(f"Save failed: {e}")
            return False

    @staticmethod
    def load(load_path: str) -> tp.Optional[tp.Any]:
        try:
            if not os.path.exists(load_path):
                tqdm.write("File not found.")
                return None
            
            with open(load_path, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except Exception as e:
            tqdm.write(f"Load failed: {e}")
            return None

    def load_json(self) -> bool:
        if not os.path.exists(self.json_filepath):
            tqdm.write(f"Error: {self.json_filepath} not exists.")
            return False
        
        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as json_file:
                self.json_data = json.load(json_file)
            return True
        except json.JSONDecodeError:
            tqdm.write("Error: incorrect JSON file format")
        except Exception as e:
            tqdm.write(f"Error: {e}")
        
        return False
    
    def run(self) -> bool:
        self.doc_title = self.json_data["title"]
        self.component_list = self.json_data["comp_data"]
        
        for component in self.component_list:
            result_component = None
            if component['type'] == "paragraph":
                result_component = self.process_text_component(component)
            elif component['type'] == "table":
                result_component = self.process_table_component(component)
            elif component['type'] == "image":
                result_component = self.process_image_component(component)
            
            if result_component is None:
                return False
            
            result_component.original_json_filepath = self.doc_title
            self.processed_components.append(result_component)
        
        return True
    
    def process_text_component(self, component) -> ProcessedComponent:
        sentence_list: tp.List[str] = self.text_segmenter.segment(component['paragraph'])
        subcomponent_embedding_requests: tp.List[EmbeddingRequestData] = [EmbeddingRequestData(sentence) for sentence in sentence_list]

        result_component = ProcessedComponent(component)
        result_component.neighbor_components = component["edge"]
        result_component.component_embedding = get_embedding(EmbeddingRequestData(component['paragraph']))
        result_component.subcomponent_embeddings = get_batch_embedding(subcomponent_embedding_requests)
        return result_component
    
    def process_table_component(self, component) -> ProcessedComponent:
        original_table: tp.List[tp.List[str]] = component["table"]
        heading_path: tp.List[str] = component["heading_path"]
        table_first_row: tp.List[str] = original_table[0]
        subcomponent_embeddings: tp.List[np.ndarray] = []
        if len(original_table) == 1:
            text, img_paths = self._flatten_table([table_first_row], heading_path)
            subcomponent_embeddings.append(get_embedding(EmbeddingRequestData(text, img_paths[0] if img_paths else "")))
        else:
            for table_line in original_table[1:]:
                text, img_paths = self._flatten_table([table_first_row, table_line], heading_path)
                subcomponent_embeddings.append(get_embedding(EmbeddingRequestData(text, img_paths[0] if img_paths else "")))
        
        result_component = ProcessedComponent(component)
        result_component.neighbor_components = component["edge"]
        result_component.subcomponent_embeddings = subcomponent_embeddings
        result_component.component_embedding = np.mean(np.stack(subcomponent_embeddings, axis=0), axis=0)
        
        return result_component
    
    def process_image_component(self, component) -> tp.Optional[ProcessedComponent]:
        full_path = get_clean_savepath_from_url(self.img_folder, component["src"])
        if not os.path.exists(full_path):
            tqdm.write(f"Error: No {full_path} exists")
            return None
        
        result_component = ProcessedComponent(component)
        result_component.neighbor_components = component["edge"]
        result_component.component_embedding = get_embedding(EmbeddingRequestData(component["caption"], full_path))
        result_component.subcomponent_embeddings = [result_component.component_embedding] # TODO

        return result_component

    def _flatten_table(self, table_data, heading_path):
        image_link_pattern = r"\[\[([^\]]+)\]\]"
        image_path_list: tp.List[str] = []
        result_text_list: tp.List[str] = []

        for table_row in table_data:
            temp_list = []
            for table_elem in table_row:
                element_img_list = [get_clean_savepath_from_url(self.img_folder, item) for item in re.findall(image_link_pattern, table_elem)]
                image_path_list.extend(element_img_list)
                text = re.sub(image_link_pattern, '', table_elem).strip()
                if text:
                    temp_list.append(text)
            result_text_list.append(" | ".join(temp_list))
        result_text = " \n ".join(result_text_list)
        
        clean_image_path_list = []
        for image_path in image_path_list:
            if os.path.exists(image_path):
                clean_image_path_list.append(image_path)
            else:
                tqdm.write(f"Error: no {image_path} exists")
        
        result_text = f"{self.doc_title} [SEP] {' > '.join(heading_path)} [SEP] {result_text}"
        
        return result_text, clean_image_path_list
    
    def _relabel(self):
        pass

class SequentialDataEmbedder:
    def __init__(
        self,
        json_folder_path: str,
        img_folder: str,
        ldoc_folder_path: str
    ) -> None:
        self.json_folder_path: str = json_folder_path
        self.img_folder: str = img_folder
        self.ldoc_folder_path: str = ldoc_folder_path
        
        self.json_path_list: tp.List[str] = []
        self.lilac_doc_dict: tp.Dict[str, LILaCDocument] = dict()
        
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.progress_bar = tqdm(total=0, desc="Embedding Parsed Data...")

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

    def load_json_filelist(self) -> bool:
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
                for edge_name in processed_comp.neighbor_components:
                    edge_range = edge_range_map.get(edge_name)
                    if edge_range:
                        id_edge_list.extend(edge_range)
                processed_comp.neighbor_components = id_edge_list

        for remapped_doc in self.lilac_doc_dict.values():
            remapped_doc.save(os.path.join(self.ldoc_folder_path, f"{remapped_doc.doc_title}.ldoc.remapped"))

        return True

    def run(self) -> bool:
        self.progress_bar.total = len(self.json_path_list)
        
        for json_path in self.json_path_list:
            new_doc = LILaCDocument(json_path, self.segmenter, self.img_folder)
            new_doc.load_json()
            new_doc_title = new_doc.json_data["title"]
            new_ldoc_path = os.path.join(self.ldoc_folder_path, f"{new_doc_title}.ldoc")
            if os.path.exists(new_ldoc_path):
                tqdm.write(f"Skip document {new_doc_title} as it is already parsed.")
                continue
            
            try:
                new_doc.run()
                new_doc.save(new_ldoc_path)
                self.lilac_doc_dict[new_doc.doc_title] = new_doc
                self.progress_bar.update(1)
            except Exception as e:
                tqdm.write(f"Skip document {new_doc_title} as it failed: {e}")
        return True

if __name__ == "__main__":
    sequential_data_embedder = SequentialDataEmbedder(PARSED_JSON_FOLDER, IMG_FOLDER, LDOC_FOLDER)
    sequential_data_embedder.load_json_filelist()
    sequential_data_embedder.run()
    
    '''
    sequential_data_embedder.edge_remapping()
    
    lilac_doc: LILaCDocument = LILaCDocument.load('/dataset/process/mmqa/Tell_Me_That_You_Love_Me,_Junie_Moon.ldoc')
    embeds = lilac_doc.processed_components[0].subcomponent_embeddings
    
    from query import get_subembeddings
    subembeddings = get_subembeddings("Which film did Ben Piazza appear in first: \"Nightwing\" or the movie that shows half of a woman's face on the poster?")
    
    emmm = get_embedding(EmbeddingRequestData("Original Poster by Saul Bass", "/dataset/crawl/mmqa_image/Tell_Me_That_You_Love_Me,_Junie_Moon_poster.jpg"))
    
    sim1 = subembeddings @ np.array(embeds).T # (Q, D) @ (D, M)
    print(sim1)
    print(np.argmax(sim1, axis=1))
    print(subembeddings @ emmm.T)
    print(np.max(sim1, axis=1))
    
    print(lilac_doc.processed_components[0].original_component)
    '''
