import os
from crawler.wiki import *

def main():
    folder_path = "./dataset/crawled/html/"
    img_folder_path = "./dataset/crawled/images/"
    wiki_batch_crawler = WikiBatchCrawler(folder_path)
    mmqa_titles = wiki_batch_crawler.get_clean_wiki_titles("dataset/MMQA_dev.jsonl", "dataset/MMQA_texts.jsonl", "dataset/MMQA_images.jsonl", "dataset/MMQA_tables.jsonl")
    
    i = 0
    for mmqa_title in mmqa_titles:
        if i%5==0:
            print(f"{i}/{len(mmqa_titles)}")
        filepath = wiki_batch_crawler.get_filepath(mmqa_title)
        wiki_crawler = WikiPage(mmqa_title, filepath)
        wiki_crawler.read_file()
        wiki_crawler.parse_lines()
        wiki_crawler.save()
        i += 1
    
    # wiki_batch_crawler.run_batch(mmqa_titles)
    
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
