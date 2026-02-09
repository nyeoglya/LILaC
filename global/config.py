import typing as tp

MMQA_PATH: str =                 "/dataset/original/mmqa/"
MMQA_IMAGE_REFERENCE_PATH: str = "/dataset/original/mmqa/final_dataset_images/"

MMQA_CRAWL_HTML_FOLDER: str =       "/dataset/crawl/mmqa_html/"
MMQA_CRAWL_IMAGE_FOLDER: str =      "/dataset/crawl/mmqa_image/"
MMQA_CRAWL_HTML_FAILED_FILE: str =  "/dataset/mmqa_html_crawl_failed.txt"
MMQA_CRAWL_IMAGE_FAILED_FILE: str = "/dataset/mmqa_image_crawl_failed.txt"

MMQA_PARSE_JSON_FOLDER: str = "/dataset/parse/mmqa_json/"
MMQA_PARSE_FAILED_FILE: str = "/dataset/parse/mmqa_parse_failed.txt"

MMQA_PROCESS_IMAGE_FOLDER: str =          "/dataset/process/mmqa_image/"
MMQA_PROCESS_IMAGE_FAILED_FILE: str =     "/dataset/process/mmqa_image_process_failed.txt"
MMQA_OBJECT_DETECT_INFO_FILE: str =       "/dataset/process/mmqa_image_object_detect.jsonl"
MMQA_IMAGE_DESCRIPTION_INFO_FILE: str =   "/dataset/process/mmqa_image_description.jsonl"
MMQA_OBJECT_DETECT_FAILED_FILE: str =     "/dataset/process/mmqa_image_object_detect_failed.txt"
MMQA_IMAGE_DESCRIPTION_FAILED_FILE: str = "/dataset/process/mmqa_image_description_failed.txt"

MMQA_LDOC_FOLDER: str = "/dataset/process/mmqa_ldoc/"
MMQA_LABELED_LDOC_FOLDER: str = "/dataset/process/labeled_mmqa_ldoc/"

MMQA_IMAGE_REFERENCE_EMBEDDING_FOR_LABELING_FILE: str =        "/dataset/process/mmqa_reference_image_embedding_for_labeling.pkl"
MMQA_IMAGE_EMBEDDING_FOR_LABELING_FILE: str =                  "/dataset/process/mmqa_image_embedding_for_labeling.pkl"
MMQA_IMAGE_REFERENCE_EMBEDDING_FOR_LABELING_FAILED_FILE: str = "/dataset/process/mmqa_reference_image_embedding_for_labeling_failed.txt"
MMQA_IMAGE_EMBEDDING_FOR_LABELING_FAILED_FILE: str =           "/dataset/process/mmqa_image_embedding_for_labeling_failed.txt"

MMQA_QUERY_CACHE_FILE: str = "/dataset/process/mmqa_query_cache.pkl"

MMQA_FINAL_GRAPH_FILENAME: str =            "/dataset/graph/mmqa.lgraph"
MMQA_GRAPH_RETRIEVAL_RESULT_FILENAME: str = "/dataset/graph/retrieval_result.jsonl"
MMQA_LLM_ANSWER_RESULT_FILE: str =          "/dataset/graph/llm_answer.jsonl"
MMQA_LLM_ANSWER_FAILED_FILE: str =          "/dataset/graph/llm_answer_failed.txt"

QWEN_SERVER_URL_LIST: tp.List[str] = ["http://lilac-qwen:8000", "http://lilac-qwen:8001", "http://lilac-qwen:8002", "http://lilac-qwen:8003"]
MMEMBED_SERVER_URL_LIST: tp.List[str] = ["http://lilac-mmembed:8000", "http://lilac-mmembed:8001", "http://lilac-mmembed:8002", "http://lilac-mmembed:8003"]

BEAM_SIZE = 30
TOP_K = 9
MAX_HOP = 100
