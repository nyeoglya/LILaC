import typing as tp

from utils import convert_image_to_png
from utils_mmqa import mmqa_get_clean_wikidocs_titles

from parser.wiki import WikiBatchParser
from crawler.wiki import WikiBatchCrawler, BatchWikiImageCrawler
# from detector import BatchObjectDetector

MMQA_PATH: str = "/dataset/original/mmqa/"
MMQA_CRAWL_HTML_FAILED_FILE: str = "/dataset/crawl/mmqa_html_failed.txt"
MMQA_CRAWL_HTML_FOLDER: str = "/dataset/crawl/mmqa_html/"
MMQA_CRAWL_IMAGE_FAILED_FILE: str = "/dataset/crawl/mmqa_image_failed.txt"
MMQA_CRAWL_IMAGE_FOLDER: str = "/dataset/crawl/mmqa_image/"

MMQA_PARSE_FAILED_FILE: str = "/dataset/parse/mmqa_failed.txt"
MMQA_PARSE_JSON_FOLDER: str = "/dataset/parse/mmqa_json/"

MMQA_OBJECT_DETECT_FAILED_FILE: str = "/dataset/parse/mmqa_detect_failed.txt"
MMQA_OBJECT_DETECT_INFO_FILE: str = "/dataset/parse/mmqa_image_detect.jsonl"

def preprocess_main() -> None:
    mmqa_wiki_doc_title_list: tp.List[str] = mmqa_get_clean_wikidocs_titles(MMQA_PATH)
    mmqa_wiki_doc_title_list = sorted(mmqa_wiki_doc_title_list)
    
    # Crawler
    '''
    batch_wiki_crawler: WikiBatchCrawler = WikiBatchCrawler(
        MMQA_CRAWL_HTML_FOLDER,
        mmqa_wiki_doc_title_list,
        "hyunseong@postech.ac.kr"
    )
    batch_wiki_crawler.run_batch(MMQA_CRAWL_FAILED_FILE)
    '''
    
    # Parser
    batch_wiki_parser: WikiBatchParser = WikiBatchParser(
        MMQA_CRAWL_HTML_FOLDER,
        MMQA_PARSE_JSON_FOLDER,
        mmqa_wiki_doc_title_list,
    )
    batch_wiki_parser.run_batch(MMQA_PARSE_FAILED_FILE)
    
    return
    
    # Image Crawler
    batch_image_crawler: BatchWikiImageCrawler = BatchWikiImageCrawler(
        MMQA_CRAWL_IMAGE_FOLDER,
        "hyunseong@postech.ac.kr"
    )
    batch_image_crawler.set_clean_imglinks_from_folder(MMQA_PARSE_JSON_FOLDER)
    batch_image_crawler.run_batch(MMQA_CRAWL_IMAGE_FAILED_FILE)
    
    
    return
    
    # Object Detector
    batch_object_detector = BatchObjectDetector(MMQA_CRAWL_IMAGE_FOLDER)
    batch_object_detector.run_batch(MMQA_OBJECT_DETECT_FAILED_FILE, MMQA_OBJECT_DETECT_INFO_FILE)
    

if __name__ == "__main__":
    preprocess_main()
