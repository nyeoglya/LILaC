from pathlib import Path

from crawler.wiki import *
from parser.wiki import *

CRAWL_HTML_FOLDER = "/dataset/crawl/mmqa_html/"
CRAWL_IMAGE_FOLDER = "/dataset/crawl/mmqa_image/"

def main():
    # Crawler
    wiki_batch_crawler = WikiBatchCrawler(CRAWL_HTML_FOLDER)
    mmqa_titles = wiki_batch_crawler.get_clean_wiki_titles("/dataset/mmqa/MMQA_dev.jsonl", "/dataset/mmqa/MMQA_texts.jsonl", "/dataset/mmqa/MMQA_images.jsonl", "/dataset/mmqa/MMQA_tables.jsonl")
    wiki_batch_crawler.run_batch(mmqa_titles)
    
    # Parser
    i = 0
    for mmqa_title in mmqa_titles:
        if i%5 == 0:
            print(f"{i}/{len(mmqa_titles)}")
        filepath = wiki_batch_crawler.get_filepath(mmqa_title)
        wiki_parser = WikiPage(mmqa_title, filepath)
        wiki_parser.read_file()
        wiki_parser.parse_lines()
        wiki_parser.save()
        i += 1
    
    # Image Crawler
    image_batch_crawler = BatchWikiImageCrawler(CRAWL_IMAGE_FOLDER)
    path = Path(CRAWL_HTML_FOLDER)
    html_json_files = [str(file) for file in path.glob("*.json")]
    img_lists = image_batch_crawler.get_clean_imglinks(html_json_files)
    image_batch_crawler.run_batch(img_lists)

if __name__ == "__main__":
    main()
