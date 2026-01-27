import os

from crawler.wiki import *
from parser.wiki import *

def main():
    crawl_folder_path = "./dataset/crawled/html/"
    img_crawl_folder_path = "./dataset/crawled/images/"
    
    # Crawler
    wiki_batch_crawler = WikiBatchCrawler(crawl_folder_path)
    mmqa_titles = wiki_batch_crawler.get_clean_wiki_titles("dataset/mmqa/MMQA_dev.jsonl", "dataset/mmqa/MMQA_texts.jsonl", "dataset/mmqa/MMQA_images.jsonl", "dataset/mmqa/MMQA_tables.jsonl")
    wiki_batch_crawler.run_batch(mmqa_titles)
    
    # Parser
    i = 0
    for mmqa_title in mmqa_titles:
        if i%5==0:
            print(f"{i}/{len(mmqa_titles)}")
        filepath = wiki_batch_crawler.get_filepath(mmqa_title)
        wiki_parser = WikiPage(mmqa_title, filepath)
        wiki_parser.read_file()
        wiki_parser.parse_lines()
        wiki_parser.save()
        i += 1
    
    # Image Crawler
    
    '''
    ref_files = []
    with open("refer.txt", "r") as file:
        for line in file:
            clean_name = line.strip().split(".")[0]
            if clean_name:
                ref_files.append(clean_name)
    
    filepath_list = [wiki_batch_crawler.get_filepath(mmqa_title) + ".json" for mmqa_title in mmqa_titles]
    image_batch_crawler = BatchWikiImageCrawler(img_folder_path)
    img_lists = image_batch_crawler.get_clean_imglinks(filepath_list)
    
    moddd = []
    for a in img_lists:
        filename = a[0]
        name_without_ext = os.path.splitext(filename)[0]
        moddd.append(name_without_ext)
    print(len(moddd), len(ref_files))
    print(sorted(list(set(ref_files) - set(moddd)))[500:510])
    '''


if __name__ == "__main__":
    main()
