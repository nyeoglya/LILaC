import os
import glob
import typing as tp

from .embed import BatchImageEmbedder
from utils.mmqa import mmqa_get_title_component_map_from_file
from config import (
    MMQA_PATH,
    MMQA_IMAGE_REFERENCE_PATH,
    MMQA_PROCESS_IMAGE_FOLDER,
    MMQA_IMAGE_EMBEDDING_FOR_LABELING_FAILED_FILE,
    MMQA_IMAGE_EMBEDDING_FOR_LABELING_FILE,
    MMQA_IMAGE_REFERENCE_EMBEDDING_FOR_LABELING_FAILED_FILE,
    MMQA_IMAGE_REFERENCE_EMBEDDING_FOR_LABELING_FILE
)

def mmqa_embedding_for_labeling():
    mmqa_component_map: tp.Dict[str, tp.Dict[str, tp.List[str]]] = mmqa_get_title_component_map_from_file(MMQA_PATH)
    reference_image_filepath_list: tp.List[str] = [datum for data in mmqa_component_map.values() for datum in data["imgid"]]
    mmqa_fullimage_map: tp.Dict[str, str] = {
        os.path.splitext(filepath)[0]: os.path.splitext(filepath)[1].lstrip(".") 
        for filepath in os.listdir(MMQA_IMAGE_REFERENCE_PATH) 
        if os.path.splitext(filepath)[1]
    }
    mmqa_reference_image_list: tp.List[str] = [
        os.path.join(MMQA_IMAGE_REFERENCE_PATH, f"{data}.{mmqa_fullimage_map[data]}")
        for data in reference_image_filepath_list
        if data in mmqa_fullimage_map
    ]
    reference_embedder: BatchImageEmbedder = BatchImageEmbedder(mmqa_reference_image_list)
    reference_embedder.run_embedding(MMQA_IMAGE_REFERENCE_EMBEDDING_FOR_LABELING_FAILED_FILE, MMQA_IMAGE_REFERENCE_EMBEDDING_FOR_LABELING_FILE)

    processed_image_filepath_list = glob.glob(os.path.join(MMQA_PROCESS_IMAGE_FOLDER, "*"))
    processed_image_embedder = BatchImageEmbedder(processed_image_filepath_list)
    processed_image_embedder.run_embedding(MMQA_IMAGE_EMBEDDING_FOR_LABELING_FAILED_FILE, MMQA_IMAGE_EMBEDDING_FOR_LABELING_FILE)
