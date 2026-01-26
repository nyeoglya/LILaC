from crawler import *
import json

def get_clean_wiki_titles(mmqa_file_path):
    titles = set()
    
    with open(mmqa_file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue # 빈 줄 건너뛰기
            
            try:
                # 한 줄씩 읽어서 딕셔너리로 변환
                entry = json.loads(line)
                
                # 만약 파일 전체가 [{}, {}] 구조라면 entry가 리스트일 수 있음
                if isinstance(entry, list):
                    data_list = entry
                else:
                    data_list = [entry]

                for item in data_list:
                    # 데이터가 문자열이면 무시 (에러 방지)
                    if not isinstance(item, dict):
                        continue
                        
                    metadata = item.get('metadata', {})
                    
                    # wiki_entities_in_answers에서 추출
                    for entity in metadata.get('wiki_entities_in_answers', []):
                        if isinstance(entity, dict):
                            title = entity.get('wiki_title')
                            if title: titles.add(title.strip())
                    
                    # wiki_entities_in_question에서 추출
                    for entity in metadata.get('wiki_entities_in_question', []):
                        if isinstance(entity, dict):
                            title = entity.get('wiki_title')
                            if title: titles.add(title.strip())

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}")
                continue

    # 공백을 언더바로 치환
    final_list = [t.replace(' ', '_') for t in titles if t]
    print(f"✅ 총 {len(final_list)}개의 고유 wiki_title을 추출했습니다.")
    return final_list

def main():
    mmqa_titles = get_clean_wiki_titles("MMQA_dev.jsonl")
    
    folder_path = "./crawled_html/"
    parser = WikiBatchParser(folder_path)
    parser.run_batch(mmqa_titles)
    
    '''
    wiki_crawler = WikiPage(folder_path + pages_to_fetch[0] + ".html")
    wiki_crawler.read_file()
    wiki_crawler.parse_lines()
    parsed_result = wiki_crawler.parsed
    for i, data in enumerate(parsed_result):
        print(f"{i}: {data}")
    '''

if __name__ == "__main__":
    main()
