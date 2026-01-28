import os

from pathlib import Path

from crawler.wiki import *
from parser.wiki import *

crawl_folder_path = "./dataset/crawl/mmqa_html/"
img_crawl_folder_path = "./dataset/crawl/mmqa_image/"

def main():
    # Crawler
    wiki_batch_crawler = WikiBatchCrawler(crawl_folder_path)
    mmqa_titles = wiki_batch_crawler.get_clean_wiki_titles("dataset/mmqa/MMQA_dev.jsonl", "dataset/mmqa/MMQA_texts.jsonl", "dataset/mmqa/MMQA_images.jsonl", "dataset/mmqa/MMQA_tables.jsonl")
    # wiki_batch_crawler.run_batch(mmqa_titles)
    
    # Parser
    '''
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
    '''
    
    # Image Crawler
    image_batch_crawler = BatchWikiImageCrawler(img_crawl_folder_path)
    path = Path(crawl_folder_path)
    html_json_files = [str(file) for file in path.glob("*.json")]
    img_lists = image_batch_crawler.get_clean_imglinks(html_json_files)
    image_batch_crawler.run_batch(img_lists)

    '''
    ref_files = []
    with open("refer.txt", "r") as file:
        for line in file:
            clean_name = line.strip().split(".")[0]
            if clean_name:
                ref_files.append(clean_name)
    
    moddd = []
    for a in img_lists:
        filename = a[0]
        filename = filename.replace('\\', '')
        name_without_ext = os.path.splitext(filename)[0]
        clean_name = name_without_ext.replace(" ", "_")
        url_encoded = urllib.parse.quote(clean_name, safe='')
        moddd.append(url_encoded)
    print(len(moddd), len(ref_files))
    dffff = sorted(list(set(ref_files) - set(moddd)))
    print(len(dffff), dffff[600:610])
    '''

if __name__ == "__main__":
    main()
