from crawler import *

def main():
    folder_path = "./crawled_html/"
    img_folder_path = "./crawled_images/"
    wiki_batch_crawler = WikiBatchCrawler(folder_path)
    mmqa_titles = wiki_batch_crawler.get_clean_wiki_titles("MMQA_dev.jsonl")
    
    '''
    wiki_batch_crawler.run_batch(mmqa_titles)
    
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
    '''
    
    filepath_list = [wiki_batch_crawler.get_filepath(mmqa_title) + ".json" for mmqa_title in mmqa_titles]
    image_batch_crawler = BatchWikiImageCrawler(img_folder_path)
    img_lists = image_batch_crawler.get_clean_imglinks(filepath_list[:10])
    print(len(img_lists))
    image_batch_crawler.run_batch(img_lists)

if __name__ == "__main__":
    main()
