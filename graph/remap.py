import os
import re
import glob
import json
import pickle
import typing as tp
from collections import Counter

from tqdm import tqdm
import numpy as np

from common import get_clean_filename, get_clean_filename_from_url
from utils.mmqa import mmqa_get_title_component_map_from_file
from embed import LILaCDocument, ProcessedComponent

SerializedList = tp.List[tp.Tuple[int, str]]

class LILaCDocMMQAMapper:
    def __init__(
        self,
        mmqa_folderpath: str,
        mmqa_ldoc_folderpath: str,
        mmqa_reference_image_embedding_filepath: str,
        mmqa_image_embedding_filepath: str
    ) -> None:
        self.mmqa_folderpath: str = mmqa_folderpath
        self.mmqa_ldoc_folderpath: str = mmqa_ldoc_folderpath
        self.mmqa_image_embedding_filepath: str = mmqa_image_embedding_filepath
        self.mmqa_reference_image_embedding_filepath: str = mmqa_reference_image_embedding_filepath
        
        self.doc_title_id_map: tp.Dict[str, tp.Dict[str, tp.List[str]]] = dict()
        self.image_embedding_map: tp.Dict[str, np.ndarray] = dict()
        self.reference_image_embedding_map: tp.Dict[str, np.ndarray] = dict()
        self.id_serialized_text_map: tp.Dict[str, str] = dict()
        self.doc_title_lilac_doc_map: tp.Dict[str, LILaCDocument] = dict()
        self.doc_title_lilac_doc_serialized_info_map: tp.Dict[str, tp.Tuple[SerializedList, SerializedList, SerializedList]] = dict()

    def load_mmqa_reference(self): # doc_title -> (serialized text, id) map
        assert os.path.exists(self.mmqa_folderpath)
        assert os.path.exists(self.mmqa_image_embedding_filepath)
        assert os.path.exists(self.mmqa_reference_image_embedding_filepath)
        
        raw_id_map = mmqa_get_title_component_map_from_file(self.mmqa_folderpath)
        self.doc_title_id_map = {get_clean_filename(title): value for title, value in raw_id_map.items()}
        with (
            open(self.mmqa_image_embedding_filepath, "rb") as mmqa_remap_image_embedding_file,
            open(self.mmqa_reference_image_embedding_filepath, "rb") as mmqa_remap_reference_image_embedding_file
        ):
            self.image_embedding_map = pickle.load(mmqa_remap_image_embedding_file)
            self.image_embedding_map = {
                os.path.splitext(os.path.basename(path))[0]: emb 
                for path, emb in self.image_embedding_map.items()
            }
            self.reference_image_embedding_map = pickle.load(mmqa_remap_reference_image_embedding_file)
            self.reference_image_embedding_map = {
                os.path.splitext(os.path.basename(path))[0]: emb
                for path, emb in self.reference_image_embedding_map.items()
            }
        
        mmqa_text_filepath = os.path.join(self.mmqa_folderpath, "MMQA_texts.jsonl")
        mmqa_table_filepath = os.path.join(self.mmqa_folderpath, "MMQA_tables.jsonl")
        
        with open(mmqa_text_filepath, "rb") as mmqa_text_file:
            for fileline in mmqa_text_file:
                line_data = json.loads(fileline)
                self.id_serialized_text_map[line_data["id"]] = self._get_clean_text(line_data["text"])
        
        with open(mmqa_table_filepath, "rb") as mmqa_table_file:
            for fileline in mmqa_table_file:
                line_data = json.loads(fileline)
                table_data = line_data["table"]["table_rows"]
                table: tp.List[tp.List[str]] = [[table_item["text"] for table_item in table_row] for table_row in table_data]
                serialized_text, _ = self._serialize_table(table)
                self.id_serialized_text_map[line_data["id"]] = self._get_clean_text(serialized_text)
    
    def load_ldoc_from_folder(self): # doc_title -> (serialized text) map
        assert os.path.exists(self.mmqa_ldoc_folderpath)
        
        for mmqa_ldoc_filepath in glob.glob(os.path.join(self.mmqa_ldoc_folderpath, "*")):
            lilac_doc = LILaCDocument.load_from_path(mmqa_ldoc_filepath)
            if not lilac_doc:
                continue
            
            serialized_text_info_list: tp.List[tp.Tuple[int, str]] = []
            serialized_table_info_list: tp.List[tp.Tuple[int, str]] = []
            image_info_list: tp.List[tp.Tuple[int, str]] = []
            
            processed_component_list: tp.List[ProcessedComponent] = lilac_doc.processed_components
            for index, processed_component in enumerate(processed_component_list):
                original_component = processed_component.original_component
                if original_component["type"] == "paragraph":
                    serialized_text_info_list.append((index, self._get_clean_text(original_component["paragraph"])))
                elif original_component["type"] == "table":
                    serialized_text, image_filepath_list = self._serialize_table(original_component["table"])
                    serialized_table_info_list.append((index, self._get_clean_text(serialized_text)))
                    image_info_list.extend([(index, filename) for filename in image_filepath_list])
                elif original_component["type"] == "image":
                    filename = get_clean_filename_from_url(original_component["src"])
                    image_info_list.append((index, filename))
            
            self.doc_title_lilac_doc_map[lilac_doc.doc_title] = lilac_doc
            self.doc_title_lilac_doc_serialized_info_map[lilac_doc.doc_title] = (serialized_text_info_list, image_info_list, serialized_table_info_list)
    
    def run_remapping(self, remapped_doc_save_folder: str): # save uuid to ldoc if possible
        assert os.path.exists(remapped_doc_save_folder)
        
        for doc_title in tqdm(self.doc_title_id_map, desc="Remapping..."):
            if doc_title not in self.doc_title_lilac_doc_serialized_info_map:
                continue
            serialized_text_info_list, image_info_list, serialized_table_info_list = self.doc_title_lilac_doc_serialized_info_map[doc_title]
            doc_component_id_info: tp.Dict[str, tp.List[str]] = self.doc_title_id_map[doc_title]
            
            remapping_data_info_list_with_none = []
            for text_component_id in doc_component_id_info["txtid"]:
                remapping_data_info_list_with_none.append(self._remap_text_component(serialized_text_info_list, text_component_id))
            for table_component_id in doc_component_id_info["tabid"]:
                remapping_data_info_list_with_none.append(self._remap_table_component(serialized_table_info_list, table_component_id))
            for image_component_id in doc_component_id_info["imgid"]:
                remapping_data_info_list_with_none.append(self._remap_image_component(image_info_list, image_component_id))
            
            remapping_data_info_list = [remapping_data_info for remapping_data_info in remapping_data_info_list_with_none if remapping_data_info]
            
            lilac_doc = self.doc_title_lilac_doc_map[doc_title]
            for index, uuid in remapping_data_info_list:
                if lilac_doc.processed_components[index].component_uuid == "": # TODO: 나중에 지우기
                    lilac_doc.processed_components[index].component_uuid = [] # TODO: 나중에 지우기
                lilac_doc.processed_components[index].component_uuid.append(uuid)
            
            lilac_doc.save_to_path(os.path.join(remapped_doc_save_folder, doc_title))
    
    def _remap_text_component(self, serialized_text_info_list: SerializedList, text_component_id: str) -> tp.Optional[tp.Tuple[int, str]]:
        if not serialized_text_info_list:
            tqdm.write(f"Error: No text in the document for {text_component_id}")
            return None
        serialized_text_data = self.id_serialized_text_map[text_component_id]
        best_item = max(
            serialized_text_info_list, 
            key=lambda item: self._recall_score(serialized_text_data, item[1])
        )
        return (best_item[0], text_component_id)
    
    def _remap_table_component(self, serialized_table_info_list: SerializedList, table_component_id: str) -> tp.Optional[tp.Tuple[int, str]]:
        serialized_text_data = self.id_serialized_text_map[table_component_id]
        if serialized_table_info_list:
            best_item = max(
                serialized_table_info_list,
                key=lambda item: self._recall_score(serialized_text_data, item[1])
            )
            return (best_item[0], table_component_id)
        else:
            tqdm.write(f"Error: No table in the document for {table_component_id}")
            return None
    
    def _remap_image_component(self, image_info_list: SerializedList, image_component_id: str) -> tp.Optional[tp.Tuple[int, str]]:
        reference_image_embedding = self.reference_image_embedding_map[image_component_id]
        valid_image_info_list = []
        for index, image_name in image_info_list:
            if image_name in self.image_embedding_map:
                valid_image_info_list.append((index, image_name))
        if valid_image_info_list:
            best_item: tp.Tuple[int, str] = max(
                valid_image_info_list,
                key=lambda item: reference_image_embedding @ self.image_embedding_map[item[1]]
            )
            return (best_item[0], image_component_id)
        else:
            tqdm.write(f"Error: No image in the document for {image_component_id}")
            return None
    
    def _get_clean_text(self, text: str) -> str:
        pattern = r"[.,!?;:\"'()\[\]{}…~\-—–_/<>《》〈〉·※ㆍ「」『』]"
        cleaned = re.sub(pattern, " ", text)
        return cleaned

    def _recall_score(self, reference_text: str, text: str) -> float:
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
    
    def _serialize_table(self, table_data: tp.List[tp.List[str]]) -> tp.Tuple[str, tp.List[str]]:
        image_link_pattern = r"\[\[([^\]]+)\]\]"
        result_text_list: tp.List[str] = []
        image_url_list: tp.List[str] = []
        for table_row in table_data:
            for table_elem in table_row:
                element_img_list = [item for item in re.findall(image_link_pattern, table_elem)]
                image_url_list.extend(element_img_list)
                text = re.sub(image_link_pattern, '', table_elem).strip()
                if text:
                    result_text_list.append(text)
        result_text = " ".join(result_text_list)
        
        image_filename_list = []
        for image_url in image_url_list:
            filename = get_clean_filename_from_url(image_url)
            image_filename_list.append(filename)
        
        return result_text, image_filename_list
