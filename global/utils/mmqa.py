import os
import json
import traceback
import typing as tp
import urllib.parse

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field
from evaluation import normalize_answer, extract_answer_from_f_call
from query import subquery_divide_query, subquery_modality_query
from config import (
    QWEN_SERVER_URL_LIST,
    MMEMBED_SERVER_URL_LIST,
    MODALITY_INSTRUCTION,
)
from common import (
    get_embedding, get_llm_response, get_query_embedding,
    EmbeddingRequestData
)

@dataclass
class MMQAQueryAnswer:
    qid: str
    question: str
    answer: tp.List[str]
    supporting_context_id: str
    supportng_context_type: str
    supporting_context: tp.Any = None
    
    llm_answer: tp.Optional[str] = None
    result_comps: tp.List[dict] = field(default_factory=list)

@dataclass
class MMQAQueryEmbedding:
    qid: str
    embedding: np.ndarray
    subcomponent_embedding_list: tp.List[np.ndarray]


def mmqa_query_eval(query_answer_list: tp.List[MMQAQueryAnswer]) -> float:
    score = 0
    total_query_len = len(query_answer_list)
    
    for query_answer in query_answer_list:
        if not query_answer.llm_answer:
            continue
        
        if "f_answer" in query_answer.llm_answer:
            extracted_list = extract_answer_from_f_call(query_answer.llm_answer)
        else:
            extracted_list = [query_answer.llm_answer]
        
        normalized_predictions = [normalize_answer(str(ans)) for ans in extracted_list]
        normalized_ground_truths = [normalize_answer(str(ans)) for ans in query_answer.answer]
        
        is_correct = False
        for pred in normalized_predictions:
            if pred in normalized_ground_truths:
                is_correct = True
                break
        
        if is_correct:
            score += 1

    em_score = score / total_query_len if total_query_len > 0 else 0
    print(f"LLM Answer Exact Match {score} among {total_query_len} queries. EM Score: {em_score:.4f}")

    return em_score

def mmqa_load_query_answer(mmqa_folderpath: str) -> tp.List[MMQAQueryAnswer]:
    dev_path: str = os.path.join(mmqa_folderpath, "MMQA_dev.jsonl")
    text_path: str = os.path.join(mmqa_folderpath, "MMQA_texts.jsonl")
    img_path: str = os.path.join(mmqa_folderpath, "MMQA_images.jsonl")
    table_path: str = os.path.join(mmqa_folderpath, "MMQA_tables.jsonl")
    
    def load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    mmqa_dev_file = load_jsonl(dev_path)
    
    mmqa_text_map = {item["id"]: item["text"] for item in load_jsonl(text_path)}
    mmqa_img_map = {item["id"]: item.get("url", item.get("path")) for item in load_jsonl(img_path)}
    mmqa_table_map = {item["id"]: item["table"] for item in load_jsonl(table_path)}
    
    result_query_answer: tp.List[MMQAQueryAnswer] = []
    
    for mmqa_line in tqdm(mmqa_dev_file, desc="Loading Query-Answer Pair..."):
        ctx_info = mmqa_line["supporting_context"][0] 
        
        new_query_answer = MMQAQueryAnswer(
            qid=mmqa_line["qid"],
            question=mmqa_line["question"],
            answer=[data["answer"] for data in mmqa_line["answers"]],
            supporting_context_id=ctx_info["doc_id"],
            supportng_context_type=ctx_info["doc_part"]
        )
        
        # 매핑 로직
        if new_query_answer.supportng_context_type == "text":
            new_query_answer.supporting_context = mmqa_text_map.get(new_query_answer.supporting_context_id)
        elif new_query_answer.supportng_context_type == "image":
            new_query_answer.supporting_context = mmqa_img_map.get(new_query_answer.supporting_context_id)
        elif new_query_answer.supportng_context_type == "table":
            new_query_answer.supporting_context = mmqa_table_map.get(new_query_answer.supporting_context_id)
        
        result_query_answer.append(new_query_answer)
    
    return result_query_answer

def mmqa_get_clean_wikidocs_titles(mmqa_folderpath: str) -> tp.List[str]:
    mmqa_devpath = os.path.join(mmqa_folderpath, "MMQA_dev.jsonl")
    mmqa_textpath = os.path.join(mmqa_folderpath, "MMQA_texts.jsonl")
    mmqa_imagepath = os.path.join(mmqa_folderpath, "MMQA_images.jsonl")
    mmqa_tablepath = os.path.join(mmqa_folderpath, "MMQA_tables.jsonl")
    
    doc_ids = set()
    img_ids = set()
    tab_ids = set()

    with open(mmqa_devpath, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            for ctx in data.get("supporting_context", []):
                d_id = ctx.get('doc_id')
                d_part = ctx.get('doc_part')
                if not d_id: continue
                    
                if d_part == "text": doc_ids.add(d_id)
                elif d_part == "image": img_ids.add(d_id)
                elif d_part == "table": tab_ids.add(d_id)
        
    final_links = set()

    def map_ids_to_urls(filepath, target_ids):
        found_urls = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # target_ids에 있는지 확인 (id가 정확히 일치해야 함)
                if item.get('id') in target_ids:
                    found_urls.append(item.get('url', ''))
        return found_urls

    final_links.update(map_ids_to_urls(mmqa_textpath, doc_ids))
    final_links.update(map_ids_to_urls(mmqa_imagepath, img_ids))
    final_links.update(map_ids_to_urls(mmqa_tablepath, tab_ids))

    clean_titles = set()
    for url in final_links:
        if not url: continue
        decoded_url = urllib.parse.unquote(url)
        title = decoded_url.replace('https://en.wikipedia.org/wiki/', '').replace(' ', '_')
        clean_titles.add(title)
            
    print(f"Extract {len(clean_titles)} unique wiki title")
    return list(clean_titles)

def mmqa_get_title_component_map_from_file(mmqa_folderpath: str) -> tp.Dict[str, tp.Dict[str, tp.List[str]]]:
    clean_titles: tp.List[str] = mmqa_get_clean_wikidocs_titles(mmqa_folderpath)
    mmqa_textpath = os.path.join(mmqa_folderpath, "MMQA_texts.jsonl")
    mmqa_imagepath = os.path.join(mmqa_folderpath, "MMQA_images.jsonl")
    mmqa_tablepath = os.path.join(mmqa_folderpath, "MMQA_tables.jsonl")

    result = {
        title: {"txtid": [], "imgid": [], "tabid": []}
        for title in clean_titles
    }

    def scan_component(filepath: str, key: str): # TODO: 이거 가져오는 로직이 뭔가 이상함. 수정해야 됨.
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                url = item.get("url")
                cid = item.get("id")
                if not url or not cid:
                    continue

                decoded = urllib.parse.unquote(url)
                title = (
                    decoded
                    .replace("https://en.wikipedia.org/wiki/", "")
                    .replace(" ", "_")
                )

                if title in result:
                    result[title][key].append(cid)

    scan_component(mmqa_textpath, "txtid")
    scan_component(mmqa_imagepath, "imgid")
    scan_component(mmqa_tablepath, "tabid")

    print(f"Extract {len(result)} wiki titles with components")
    return result

def get_mmqa_subquery_and_subembedding_list(embedding_server_url: str, llm_server_url: str, query_text: str) -> tp.Tuple[tp.List[str], np.ndarray]:
    query: str = subquery_divide_query(query_text)
    subquery_response: str = get_llm_response(embedding_server_url, query)
    subquery_list: tp.List[str] = subquery_response.replace("\n","").split(";")
    cleaned_subquery_list: tp.List[str] = [s.strip() for s in subquery_list if s.strip()]
    
    embedding_list: tp.List[np.ndarray] = []
    for cleaned_subquery in cleaned_subquery_list:
        raw_modality: str = get_llm_response(embedding_server_url, subquery_modality_query(cleaned_subquery))
        modality_key: str = raw_modality.strip().lower().replace(".", "")
        modality_instruction: str = MODALITY_INSTRUCTION.get(modality_key, MODALITY_INSTRUCTION["text"])
        result_embedding: np.ndarray = get_query_embedding(llm_server_url, modality_instruction, cleaned_subquery, "")
        embedding_list.append(result_embedding)
        
    return cleaned_subquery_list, np.stack(embedding_list) if embedding_list else np.array([])

def mmqa_cache_query_process(mmqa_path: str, llm_server_url: str, embedding_server_url: str, failed_filepath: str, result_filepath: str) -> bool:
    query_answer_list: tp.List[MMQAQueryAnswer] = mmqa_load_query_answer(mmqa_path)
    with open(result_filepath, 'a', encoding='utf-8') as result_file:
        for query_answer_data in tqdm(query_answer_list, desc="Caching Subquery and Query Embedding..."):
            try:
                query_embedding: np.ndarray = get_embedding(embedding_server_url, EmbeddingRequestData(query_answer_data.question))
                subquery_list, subembedding_list = get_mmqa_subquery_and_subembedding_list(embedding_server_url, llm_server_url, query_answer_data.question)
                record: tp.Dict[str, tp.Any] = {
                    'qid': query_answer_data.qid,
                    'query': query_answer_data.question,
                    'subqueries': subquery_list,
                    'embedding': query_embedding.tolist(),
                    'subembedding_with_modality': subembedding_list.tolist(),
                }
                result_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                result_file.flush()
            except Exception as e:
                traceback.print_exc()
                with open(failed_filepath, "a") as error_file:
                    error_file.write(f"Error on QID: {query_answer_data.qid}, Error: {str(e)}\n")
    return True

def mmqa_load_cached_query_data(data_filepath: str) -> tp.List[MMQAQueryEmbedding]:
    query_embedding_list: tp.List[MMQAQueryEmbedding] = []
    with open(data_filepath, 'r', encoding='utf-8') as data_file:
        for data_line in tqdm(data_file, desc="Loading Cached Query Data..."):
            json_data = json.loads(data_line)
            new_mmqa_query_embedding = MMQAQueryEmbedding(
                qid=json_data["qid"],
                embedding=json_data["embedding"],
                subcomponent_embedding_list=[np.array(subcomponent_embedding) for subcomponent_embedding in json_data["subembedding_with_modality"]]
            )
            query_embedding_list.append(new_mmqa_query_embedding)
    return query_embedding_list
