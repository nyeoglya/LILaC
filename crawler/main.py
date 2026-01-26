from crawler import *

def main():
    wiki_crawler = WikiPage("9K33 Osa")
    wiki_crawler.run()
    parsed_result = wiki_crawler.parsed
    for i, data in enumerate(parsed_result):
        print(f"{i}: {data}")
    
if __name__ == "__main__":
    main()
    
    '''
    parser = WikiBatchParser(lang="en")
    pages_to_fetch = ["9K33_Osa", "MIM-104_Patriot", "S-300_missile_system"]
    
    final_data = parser.run_batch(pages_to_fetch)
    
    for data in final_data:
        print(f"Page: {data['page']} | Images found: {len(data['images'])}")
    '''
