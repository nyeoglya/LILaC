import json
import typing as tp

from .utils import QueryAnswer

def mmqa_load(dev_path: str, text_path: str, img_path: str, table_path: str) -> tp.List[QueryAnswer]:
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
    
    result_query_answer: tp.List[QueryAnswer] = []
    
    for mmqa_line in mmqa_dev_file:
        ctx_info = mmqa_line["supporting_context"][0] 
        
        new_query_answer = QueryAnswer(
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


if __name__ == "__main__":
    result_query_answer = mmqa_load(
        "/dataset/mmqa/MMQA_dev.jsonl",
        "/dataset/mmqa/MMQA_texts.jsonl",
        "/dataset/mmqa/MMQA_images.jsonl",
        "/dataset/mmqa/MMQA_tables.jsonl",
    )
