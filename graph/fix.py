import os

from processor import *

def global_remapping_and_verify(ldoc_folder_path, save_folder_path):
    # 모든 .ldoc 파일 목록 가져오기 및 정렬
    filenames = [f for f in os.listdir(ldoc_folder_path) if f.endswith(".ldoc")]
    filenames.sort() 

    all_docs = []
    
    # 모든 파일 메모리로 로드
    print(f"[Step 1] Loading {len(filenames)} files")
    for filename in filenames:
        file_path = os.path.join(ldoc_folder_path, filename)
        doc = LILaCDocument.load(file_path)
        if doc:
            all_docs.append(doc)

    # 글로벌 ID 부여 (Edge 리매핑 제외)
    print(f"[Step 2] Re-indexing IDs sequentially")
    current_global_id = 0
    for doc in all_docs:
        for comp in doc.processed_components:
            comp.id = current_global_id
            current_global_id += 1

    # 결과 저장 (.ldoc.idmap 확장자)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    print(f"[Step 3] Saving files with .idmap extension")
    for doc in all_docs:
        # 기존 확장자(.ldoc)를 유지하면서 .idmap을 붙이거나 교체
        save_name = f"{doc.doc_title}.ldoc.idmap"
        doc.save(os.path.join(save_folder_path, save_name))

    print(f"Successfully saved {len(all_docs)} files.")

    # 검증 (Verification)
    print(f"[Step 4] Verifying ID continuity")
    all_ids = []
    for doc in all_docs:
        for comp in doc.processed_components:
            all_ids.append(comp.id)

    is_valid = True
    if not all_ids:
        print("Error: No IDs found to verify.")
        return

    min_id = min(all_ids)
    max_id = max(all_ids)
    unique_count = len(set(all_ids))
    is_continuous = (max_id - min_id + 1 == len(all_ids))

    print(f"  > Total components indexed: {len(all_ids)}")
    print(f"  > ID Range: {min_id} ~ {max_id}")
    
    if min_id == 0 and is_continuous and unique_count == len(all_ids):
        print("Verification Passed: All IDs are sequential and unique (0 to N-1).")
    else:
        is_valid = False
        print("Verification Failed!")
        if min_id != 0: print(f"    - Starting ID is {min_id}, not 0.")
        if unique_count != len(all_ids): print(f"    - Found {len(all_ids) - unique_count} duplicate IDs.")
        if not is_continuous: print("    - There are gaps in the ID sequence.")

    return is_valid

if __name__ == "__main__":
    SOURCE_FOLDER = "/dataset/process/mmqa/"
    TARGET_FOLDER = "/dataset/process/mmqa_idmapped/"
    
    global_remapping_and_verify(SOURCE_FOLDER, TARGET_FOLDER)
