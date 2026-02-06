import os
import re
import json
import pickle
import unicodedata
import typing as tp
from collections import Counter

from tqdm import tqdm
import numpy as np

import pysbd

from config import (
    MMQA_PATH,
    MMEMBED_SERVER_URL_LIST,
    MMQA_PARSE_JSON_FOLDER,
    MMQA_LDOC_FOLDER,
    MMQA_LDOC_FOLDER_TEMP,
    MMQA_IMAGE_DESCRIPTION_INFO_FILE,
    MMQA_OBJECT_DETECT_INFO_FILE,
    MMQA_PROCESS_IMAGE_FOLDER,
    MMQA_REMAP_IMAGE_EMBEDDING_PT,
    MMQA_REMAP_REFERENCE_EMBEDDING_PT
)
from common import (
    get_embedding, get_clean_filename_from_url, get_clean_savepath,
    EmbeddingRequestData
)
from base_data_structure import ComponentData

class ProcessedComponent:
    def __init__(self, original_component) -> None:
        self.component_uuid: str = "" # unique id for calculating retrieval score (ex: mmqa cid)
        self.doc_title: str = ""
        self.original_component: ComponentData = original_component
        self.component_embedding: np.ndarray = np.array([])
        self.subcomponent_embeddings: tp.List[np.ndarray] = [] # list(subcomp embed vector)
        self.neighbor_components: tp.List[str] = [] # list(doc title)

class LILaCDocument:
    def __init__(
        self,
        text_segmenter,
        image_metadata_map: tp.Dict[str, tp.Dict],
        processed_image_folder: str
    ) -> None:
        self.processed_image_folder: str = processed_image_folder
        self.image_metadata_map: tp.Dict[str, tp.Dict] = image_metadata_map
        self.text_segmenter = text_segmenter

        self.doc_title: str = ""
        self.original_json_data: tp.Dict = dict()
        
        self.processed_components: tp.List[ProcessedComponent] = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state["image_metadata_map"] = None
        state["text_segmenter"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save_to_path(self, save_filepath: str) -> bool:
        try:
            with open(save_filepath, 'wb') as f:
                pickle.dump(self, f)
            tqdm.write(f"Successfully saved to {save_filepath}")
            return True
        except Exception as e:
            tqdm.write(f"Save failed: {e}")
            return False

    @staticmethod
    def load_from_path(ldoc_filepath: str) -> tp.Optional["LILaCDocument"]:
        try:
            if not os.path.exists(ldoc_filepath):
                tqdm.write("File not found.")
                return None
            
            with open(ldoc_filepath, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except Exception as e:
            tqdm.write(f"Load failed: {e}")
            return None

    def load_json(self, json_filepath: str) -> bool:
        if not os.path.exists(json_filepath):
            tqdm.write(f"Error: {json_filepath} not exists.")
            return False
        
        try:
            with open(json_filepath, 'r', encoding='utf-8') as json_file:
                self.original_json_data = json.load(json_file)
                self.doc_title = self.original_json_data["title"]
            return True
        except Exception as e:
            tqdm.write(f"Error: {e}")
        
        return False
    
    def run_embedding(self) -> bool:
        self.doc_title = self.original_json_data["title"]
        self.component_list = self.original_json_data["comp_data"]
        
        for component in self.component_list:
            result_component = None
            if component['type'] == "paragraph":
                result_component = self.process_text_component(component)
            elif component['type'] == "table":
                result_component = self.process_table_component(component)
            elif component['type'] == "image":
                result_component = self.process_image_component(component)
            
            if result_component is None:
                continue
            
            result_component.doc_title = self.doc_title
            self.processed_components.append(result_component)
        
        return True
    
    def process_text_component(self, component) -> ProcessedComponent:
        serialized_text_prefix = f"{self.doc_title} [SEP] {' , '.join(component['heading_path'])} [SEP] "
        sentence_list: tp.List[str] = self.text_segmenter.segment(component['paragraph'])

        result_component = ProcessedComponent(component)
        result_component.neighbor_components = component["edge"]
        serialized_text = serialized_text_prefix + component['paragraph']
        result_component.component_embedding = get_embedding(MMEMBED_SERVER_URL_LIST[0], EmbeddingRequestData(serialized_text.replace("\n","")))
        result_component.subcomponent_embeddings = [get_embedding(MMEMBED_SERVER_URL_LIST[0], EmbeddingRequestData(serialized_text_prefix + sentence.replace("\n",""))) for sentence in sentence_list]
        
        return result_component
    
    def process_table_component(self, component) -> ProcessedComponent:
        serialized_text_prefix = f"{self.doc_title} [SEP] {' , '.join(component['heading_path'])} [SEP] "
        
        original_table: tp.List[tp.List[str]] = component["table"]
        table_first_row: tp.List[str] = original_table[0]
        subcomponent_embeddings: tp.List[np.ndarray] = []
        
        if len(original_table) <= 2:
            text, image_filepath_list = self._flatten_table(original_table)
            serialized_text = serialized_text_prefix + text
            subcomponent_embeddings = [get_embedding(MMEMBED_SERVER_URL_LIST[0], EmbeddingRequestData(serialized_text.replace("\n", ""), image_filepath_list[0] if image_filepath_list else ""))]
        else:
            for table_line in original_table[1:]:
                text, image_filepath_list = self._flatten_table([table_first_row, table_line])
                serialized_text = serialized_text_prefix + text
                subcomponent_embeddings.append(get_embedding(MMEMBED_SERVER_URL_LIST[0], EmbeddingRequestData(serialized_text.replace("\n", ""), image_filepath_list[0] if image_filepath_list else "")))
        
        result_component = ProcessedComponent(component)
        result_component.neighbor_components = component["edge"]
        result_component.subcomponent_embeddings = subcomponent_embeddings
        full_text, full_img_filepath_list = self._flatten_table(original_table)
        result_component.component_embedding = get_embedding(MMEMBED_SERVER_URL_LIST[0], EmbeddingRequestData(full_text.replace("\n", ""), full_img_filepath_list[0] if full_img_filepath_list else "")) # np.mean(np.stack(subcomponent_embeddings, axis=0), axis=0) # TODO
        
        return result_component
    
    def process_image_component(self, component) -> tp.Optional[ProcessedComponent]:
        clean_imagename, _ = get_clean_filename_from_url(component["src"])
        clean_png_image_filepath = get_clean_savepath(self.processed_image_folder, clean_imagename, "png")
        if not os.path.exists(clean_png_image_filepath):
            tqdm.write(f"Error: No {clean_png_image_filepath} exists")
            return None
        
        image_metadata = self.image_metadata_map[clean_imagename + ".png"]
        serialized_text = f"{self.doc_title} [SEP] {' , '.join(component['heading_path'])} [SEP] {component['caption']}"
        serialized_text += f" [SEP] {image_metadata['explanation']}" if image_metadata["explanation"] else ""
        serialized_text += f" [SEP] {image_metadata['ocr']}" if image_metadata["ocr"] else ""
        
        result_component = ProcessedComponent(component)
        result_component.neighbor_components = component["edge"]
        
        result_component.component_embedding = get_embedding(MMEMBED_SERVER_URL_LIST[0], EmbeddingRequestData(serialized_text, clean_png_image_filepath))
        if not image_metadata["bboxes"]:
            result_component.subcomponent_embeddings = [result_component.component_embedding]
        else:
            for bounding_box in image_metadata["bboxes"]:
                result_component.subcomponent_embeddings.append(get_embedding(MMEMBED_SERVER_URL_LIST[0], EmbeddingRequestData(serialized_text, clean_png_image_filepath, bounding_box)))
        
        return result_component

    def _flatten_table(self, table_data: tp.List) -> tp.Tuple[str, tp.List[str]]:
        image_link_pattern = r"\[\[([^\]]+)\]\]"
        image_url_list: tp.List[str] = []
        result_text_list: tp.List[str] = []

        for table_row in table_data:
            temp_list = []
            for table_elem in table_row:
                element_img_list = [item for item in re.findall(image_link_pattern, table_elem)]
                image_url_list.extend(element_img_list)
                text = re.sub(image_link_pattern, '', table_elem).strip()
                if text:
                    temp_list.append(text)
            result_text_list.append(" ".join(temp_list))
        result_text = " [SEP] ".join(result_text_list)
        
        clean_imagename_list = []
        ind = 1
        for image_url in image_url_list:
            clean_imagename, _ = get_clean_filename_from_url(image_url)
            clean_png_imagepath = get_clean_savepath(self.processed_image_folder, clean_imagename, "png")
            if os.path.exists(clean_png_imagepath):
                clean_imagename_list.append(clean_imagename)
                result_text += f" [SEP] <Image {ind}> {clean_imagename} {self.image_metadata_map[clean_imagename+'.png']['explanation']}"
                ind += 1
        return result_text, clean_imagename_list

class SequentialDataEmbedder:
    def __init__(
        self,
        json_folderpath: str,
        ldoc_folderpath: str,
        image_description_filepath: str,
        image_object_detection_filepath: str,
        processed_image_folderpath: str,
    ) -> None:
        self.json_folderpath: str = json_folderpath
        self.ldoc_folderpath: str = ldoc_folderpath
        self.image_description_filepath: str = image_description_filepath
        self.image_object_detection_filepath: str = image_object_detection_filepath
        self.processed_image_folderpath: str = processed_image_folderpath
        
        self.json_path_list: tp.List[str] = []
        self.lilac_doc_dict: tp.Dict[str, LILaCDocument] = dict()
        self.image_metadata_map: tp.Dict[str, tp.Dict] = dict()
        
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.progress_bar = tqdm(total=0, desc="Embedding Parsed Data...")

    def load_image_infodata(self):
        assert os.path.exists(self.image_description_filepath)
        assert os.path.exists(self.image_object_detection_filepath)
        
        def clean_ocr_text(text: str) -> str:
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        with open(self.image_description_filepath, "r", encoding="utf-8") as image_object_detection_file:
            for image_description_line in image_object_detection_file:
                image_description = json.loads(image_description_line)
                self.image_metadata_map[image_description["image"]] = { # TODO: file_path 출력 고치기
                    "explanation": image_description["explanation"],
                    "ocr": clean_ocr_text(image_description["ocr"]),
                    "bboxes": []
                }
        
        with open(self.image_object_detection_filepath, "r", encoding="utf-8") as image_object_detection_file:
            for image_object_detection_line in image_object_detection_file:
                image_object_detection = json.loads(image_object_detection_line)
                if image_object_detection["image"] in self.image_metadata_map:
                    self.image_metadata_map[image_object_detection["image"]]["bboxes"] = image_object_detection["bboxes"]
                else:
                    tqdm.write(f"ERROR: not found object detection data of {image_object_detection['image']}")
    
    
    def load_ldoc(self) -> bool:
        assert os.path.exists(self.ldoc_folderpath)
        self.load_image_infodata()

        self.lilac_doc_dict = dict()
        for filename in os.listdir(self.ldoc_folderpath):
            if filename.endswith(".ldoc"):
                file_path = os.path.join(self.ldoc_folderpath, filename)
                new_doc: tp.Optional[LILaCDocument] = LILaCDocument.load_from_path(file_path)
                if new_doc:
                    self.lilac_doc_dict[new_doc.doc_title] = new_doc

        return True

    def load_json_filelist(self) -> bool:
        assert os.path.exists(self.json_folderpath)
        self.load_image_infodata()

        self.json_path_list = []
        for filename in os.listdir(self.json_folderpath):
            if filename.endswith(".json"):
                file_path = os.path.join(self.json_folderpath, filename)
                self.json_path_list.append(file_path)

        return True

    def run_embedding(self) -> bool:
        self.progress_bar.total = len(self.json_path_list)
        
        for json_path in self.json_path_list:
            new_ldoc: LILaCDocument = LILaCDocument(self.segmenter, self.image_metadata_map, self.processed_image_folderpath)
            new_ldoc.load_json(json_path)
            new_ldoc_filepath = os.path.join(self.ldoc_folderpath, f"{new_ldoc.doc_title}.ldoc")
            if os.path.exists(new_ldoc_filepath):
                tqdm.write(f"Skip document {new_ldoc.doc_title} as it is already parsed.")
                continue
            
            try:
                new_ldoc.run_embedding()
                new_ldoc.save_to_path(new_ldoc_filepath)
                self.lilac_doc_dict[new_ldoc.doc_title] = new_ldoc
                self.progress_bar.update(1)
            except Exception as e:
                print(f"Skip document {new_ldoc.doc_title} as it failed: {e}")
        return True
    
    '''
    def process_single_json(self, json_path: str, llm_server: str) -> bool:
        new_ldoc: LILaCDocument = LILaCDocument(self.segmenter, self.image_metadata_map, self.processed_image_folderpath)
        new_ldoc.load_json(json_path)
        new_ldoc_filepath = os.path.join(self.ldoc_folderpath, f"{new_ldoc.doc_title}.ldoc")
        if os.path.exists(new_ldoc_filepath):
            tqdm.write(f"Skip document {new_ldoc.doc_title} as it is already parsed.")
            return True
            
        try:
            new_ldoc.run_embedding()
            new_ldoc.save_to_path(new_ldoc_filepath)
            self.lilac_doc_dict[new_ldoc.doc_title] = new_ldoc
            return True
        except Exception as e:
            print(f"Skip document {new_ldoc.doc_title} as it failed: {e}")
            return False
    
    def run_batch_embedding(self, failed_file_path: str, embed_server_list: tp.List[str]) -> bool:
        assert os.path.exists(self.json_folderpath)
        assert os.path.exists(self.ldoc_folderpath)
        
        self.progress_bar.total = len(self.json_path_list)
        server_cycle = itertools.cycle(embed_server_list)

        with open(failed_file_path, 'a', encoding='utf-8') as failed_file, \
            ThreadPoolExecutor(max_workers=len(embed_server_list)) as executor:

            future_to_path = {
                executor.submit(
                    self.process_single_json,
                    image_path,
                    next(server_cycle)
                ): image_path
                for image_path in self.json_path_list
            }

            for _ in as_completed(future_to_path):
                self.progress_bar.update(1)

        return True
    '''

class LILaCDocMMQAMapper:
    def __init__(
        self,
        mmqa_folderpath: str,
        mmqa_remap_reference_embedding_filepath: str,
        mmqa_remap_image_embedding_filepath: str,
        mmqa_ldoc_folderpath: str,
    ) -> None:
        self.mmqa_folderpath: str = mmqa_folderpath
        self.mmqa_remap_image_embedding_filepath: str = mmqa_remap_image_embedding_filepath
        self.mmqa_remap_reference_embedding_filepath: str = mmqa_remap_reference_embedding_filepath
        self.mmqa_ldoc_folderpath: str = mmqa_ldoc_folderpath
        
        self.doc_title_id_map: tp.Dict[str, tp.Tuple[str, ComponentData]] = dict()

    def load_mmqa_reference(self): # doc_title -> (serialized text, id) map
        assert os.path.exists(self.mmqa_folderpath)
        assert os.path.exists(self.mmqa_ldoc_folderpath)
        assert os.path.exists(self.mmqa_remap_image_embedding_filepath)
        assert os.path.exists(self.mmqa_remap_reference_embedding_filepath)
        
        self.doc_title_id_map: tp.Dict[str, tp.Tuple[str, ComponentData]] = dict()
        with (
            open(self.mmqa_remap_image_embedding_filepath, "w", encoding="utf-8") as mmqa_remap_image_embedding_file,
            open(self.mmqa_remap_reference_embedding_filepath, "w", encoding="utf-8") as mmqa_remap_reference_embedding_file,
            open(os.path.join(self.mmqa_folderpath, "mmqa_text.jsonl"), "w", encoding="utf-8") as mmqa_text_file
        ):
            pass
    
    def load_ldoc_from_folder(self): # doc_title -> (serialized text) map
        pass
    
    def run_remapping(self): # save uuid to ldoc if possible
        pass

    def recall_score_with_cleanedtext(self, reference_text: str, text: str) -> float:
        def remove_punctuation(text: str) -> str:
            pattern = r"[.,!?;:\"'()\[\]{}…~\-—–_/<>《》〈〉·※ㆍ「」『』]"
            cleaned = re.sub(pattern, "", text)
            return cleaned
        return self.recall_score(remove_punctuation(reference_text), remove_punctuation(text))

    def recall_score(self, reference_text: str, text: str) -> float:
        ref_tokens = reference_text.split()
        pred_tokens = text.split()

        if not ref_tokens:
            return 0.0
        
        ref_counter = Counter(ref_tokens)
        pred_counter = Counter(pred_tokens)
        
        overlap = 0
        for token, ref_count in ref_counter.items():
            overlap += min(ref_count, pred_counter.get(token, 0))
        
        recall = overlap / len(ref_tokens)
        return recall

def process_main():
    # Data Embedder
    sequential_data_embedder = SequentialDataEmbedder(
        MMQA_PARSE_JSON_FOLDER,
        MMQA_LDOC_FOLDER,
        MMQA_IMAGE_DESCRIPTION_INFO_FILE,
        MMQA_OBJECT_DETECT_INFO_FILE,
        MMQA_PROCESS_IMAGE_FOLDER
    )
    sequential_data_embedder.load_json_filelist()
    sequential_data_embedder.run_embedding()
    
    lilac_data_mmqa_mapper = LILaCDocMMQAMapper(
        MMQA_PATH,
        MMQA_REMAP_REFERENCE_EMBEDDING_PT,
        MMQA_REMAP_IMAGE_EMBEDDING_PT,
        MMQA_LDOC_FOLDER_TEMP
    )
    lilac_data_mmqa_mapper.load_mmqa_reference()
    lilac_data_mmqa_mapper.load_ldoc_from_folder()
    lilac_data_mmqa_mapper.run_remapping()
    
    
    

if __name__ == "__main__":
    process_main()
    
    # ldoc = LILaCDocument.load_from_path('/dataset/process/mmqa_ldoc/Claire_Coffee.ldoc')
