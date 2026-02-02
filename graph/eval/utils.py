import re
import string
import typing as tp

from dataclasses import dataclass, field


@dataclass
class QueryAnswer:
    qid: str
    question: str
    answer: tp.List[str]
    supporting_context_id: str
    supportng_context_type: str
    supporting_context: tp.Any = None
    
    llm_answer: tp.Optional[str] = None
    result_comps: tp.List[dict] = field(default_factory=list)


def extract_answer_from_f_call(llm_output: str) -> list[str]:
    pattern = r'f_answers\s*\(\s*\[(.*?)\]\s*\)'
    match = re.search(pattern, llm_output, re.DOTALL)
    
    if match:
        content = match.group(1)
        answers = re.findall(r'["\'](.*?)["\']', content)
        return [ans.strip() for ans in answers]
    
    return [llm_output.strip()]

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def query_eval(query_answer_list: tp.List[QueryAnswer]) -> float:
    score = 0
    total_query_len = len(query_answer_list)
    
    for query_answer in query_answer_list:
        if not query_answer.llm_answer:
            continue
        
        if "f_answer" in query_answer.llm_answer:
            extracted_list = extract_answer_from_f_call(query_answer.llm_answer)
        else:
            extracted_list = [query_answer.llm_answer]
        
        normalized_predictions = [normalize_answer(ans) for ans in extracted_list]
        normalized_ground_truths = [normalize_answer(text) for text in query_answer.answer]
        
        is_correct = False
        for pred in normalized_predictions:
            if pred in normalized_ground_truths:
                is_correct = True
                break
        
        if is_correct:
            score += 1
    
    em_score = score * 100 / total_query_len if total_query_len > 0 else 0
    print(f"LLM Answer Exact Match {score} among {total_query_len} queries. EM Score: {em_score:.2f}")

    return em_score
