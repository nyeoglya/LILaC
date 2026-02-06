import os
import typing as tp

from graph import LILaCDocument, ProcessedComponent
from utils.mmqa import MMQAQueryAnswer

def mmqa_embed_test(query_answer_list: tp.List[MMQAQueryAnswer], ldoc_folderpath: str):
    query_list = [query_answer.question for query_answer in query_answer_list]
    collision_samples = []
    i = 0
    for ldoc_filename in os.listdir(ldoc_folderpath):
        ldoc_filepath = os.path.join(ldoc_folderpath, ldoc_filename)
        ldoc = LILaCDocument.load_from_path(ldoc_filepath)
        if not ldoc:
            print("continue")
            continue
        for processed_component in ldoc.processed_components:
            if processed_component.component_uuid:
                i += len(processed_component.component_uuid)
                if len(processed_component.component_uuid) > 1:
                    collision_samples.append({
                        'type': processed_component.original_component['type'],
                        'uuids': processed_component.component_uuid,
                        'content_preview': str(processed_component.original_component.get('paragraph', ''))[:30]
                    })
            # processed_component.component_embedding

    for s in collision_samples:
        print(f"[{s['type']}] 중복 ID: {s['uuids']} | 내용: {s['content_preview']}...")
