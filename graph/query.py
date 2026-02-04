import numpy as np
import typing as tp

from utils import (
    get_clean_savepath_from_url, get_query_embedding, get_llm_response
)

def get_subembeddings(text, img_path="") -> np.ndarray:
    query = subquery_divide_query(text)
    subqueries = get_llm_response("", query).replace("\n","").split(";")
    embeddings = []
    for subquery in subqueries:
        modality = get_llm_response("", subquery_modality_query(subquery))
        result_embedding = get_query_embedding(modality, subquery, img_path)
        embeddings.append(result_embedding)
    return np.stack(embeddings)

SUBQUERY_DIVIDE_QUERY = """Instruction: You are a retrieval-oriented query decomposer.

Goal – Produce the smallest set (1 – 5) of component-targeting sub-queries. Each sub-query must describe one retrievable component (sentence, paragraph, table row, figure, etc.) whose embedding should be matched. Together, the sub-queries must supply all the information needed to answer the original question.

Guidelines:
1. Entity & noun-phrase coverage: Every noun phrase and named entity that appears in the original question must appear at least once across the entire set of sub-queries (you may distribute them). Keep each phrase exactly as written.
2. One-component rule: A sub-query should reference only the facts expected to co-occur within the same component. If two facts will likely be in different components, put them in different sub-queries.
3. No unnecessary splitting: If the whole answer can be found in a single component, return only one sub-query.
4. De-contextualize: Rewrite pronouns and implicit references so every sub-query is understandable on its own.
5. Keyword distribution: Spread constraints logically (e.g., one sub-query for “light rail completion date”, another for “city with a large arched bridge from the 1997 Australia rugby-union test match”).
6. Remove redundancy: Merge duplicate or paraphrased sub-queries before you output.
7. Ordering for dependencies: If the answer to one sub-query is needed for another, place the prerequisite first.
8. Output format: Return the sub-queries separated by a semicolon (;). Do not provide any keys, explanations, introduction, or extra text. Ensure each sub-query is on a new line for clarity.

Question: {query}
Output: """

SUBQUERY_MODALITY_QUERY = """Instruction: You are a modality selector for multimodal QA.

Task: Given the single sub-question below, choose the one modality that is most appropriate for obtaining its answer.

Allowed modalities:
* text: unstructured prose (paragraphs, sentences, propositions)
* table: structured rows/columns (spreadsheets, stats tables, infoboxes)
* image: visual information (photos, posters, logos, charts)

Heuristics:
1. Numeric totals, percentages, year-by-year figures → table
2. Visual appearance, colours, logos, “what does . . . look like” → image
3. Definitions, roles, biographies, causal explanations, quotes → text
4. If two modalities could work, pick the one that will yield the answer fastest.

Output format: Return only the modality label on a single line – exactly text, table, or image. No JSON, no additional text.

Subquery: {subquery}
Output: """

LLM_QUESTION_QUERY = """Instruction: Using the f_answers() API, return a list of answers to the question based on retrieved webpage components. A retrieved component can be a passage, a table, or an image. Strictly follow the format of the example below and keep the answer short. For yes/no questions, respond only with f_answers(["yes"]) or f_answers(["no"]).

Example:
[Passage] Document title: South Asia The current territories of Afghanistan, Bangladesh, Bhutan, Maldives, Nepal, India, Pakistan, and Sri Lanka form South Asia. The South Asian Association for Regional Cooperation (SAARC) is an economic cooperation organisation established in 1985 that includes all eight nations comprising South Asia.
[Passage] Document title: UK Joint Expeditionary Force The UK Joint Expeditionary Force (JEF) is a United Kingdom-led expeditionary force which may include Denmark, Finland, Estonia, Latvia, Lithuania, the Netherlands, Sweden, and Norway. It is distinct from the Franco-British Combined Joint Expeditionary Force.
[Table] Document title: Lithuanian Armed Forces — Current operations Deployment | Organization | Operation | Personnel Somalia | EU | Operation Atalanta | 15 Mali | EU | EUTM Mali | 2 Afghanistan | NATO | Operation Resolute Support | 29 Libya | EU | EU Navfor Med | 3 Mali | UN | MINUSMA | 39 Iraq | CJTF | Operation Inherent Resolve | 6 Central African Republic | EU | EUFOR RCA | 1 Kosovo | NATO | KFOR | 1 Ukraine | — | Training mission | 40
Question: Among the Lithuanian Armed Forces’ current operations, which deployment involves fewer personnel: Kosovo, or the deployment in the nation that, along with six others, constitutes the sub-continent of South Asia? Answer: The South Asia passage shows Afghanistan is part of that region. The table lists 29 personnel in Afghanistan and only 1 in Kosovo, so f_answers(["Kosovo"]).

Using the images and texts given, answer the question below in a single word or phrase.
{retrieved_comp_text}
Question: {question}
Answer: """

def subquery_divide_query(query: str) -> str:
    return SUBQUERY_DIVIDE_QUERY.format(query)

def subquery_modality_query(subquery: str) -> str:
    return SUBQUERY_MODALITY_QUERY.format(subquery=subquery)

def llm_question_query(question: str, img_folder: str, doc_list: list[str], comp_list: list[dict]) -> tp.Tuple[str, tp.List[str]]:
    retrieved_comp_list = []
    img_paths = []
    for doc_title, comp in zip(doc_list, comp_list):
        if comp["type"] == "paragraph":
            retrieved_comp_list.append(f"[Passage] Document title: {doc_title} {comp['paragraph']}")
        elif comp["type"] == "table":
            table_text = " | ".join([" | ".join(row) for row in comp["table"]])
            retrieved_comp_list.append(f"[Table] Document title: {doc_title} {table_text}")
        elif comp["type"] == "image":
            img_paths.append(get_clean_savepath_from_url(img_folder, comp["src"]))
    
    retrieved_comp_text = "\n".join(retrieved_comp_list)
    return LLM_QUESTION_QUERY.format(question=question, retrieved_comp_text=retrieved_comp_text), img_paths
