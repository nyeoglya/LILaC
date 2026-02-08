import re
import string
import typing as tp

def extract_answer_list_from_f_call(llm_output: str) -> tp.List[str]:
    pattern = r'f_answers\s*\(\s*\[(.*?)\]\s*\)'
    match = re.search(pattern, llm_output, re.DOTALL)
    
    if match:
        content = match.group(1)
        answers = re.findall(r'["\'](.*?)["\']', content)
        return [ans.strip() for ans in answers]
    
    return [llm_output.strip()]

def normalize_answer(input_string: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(input_string))))
