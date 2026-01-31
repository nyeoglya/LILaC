from utils import *

def get_subembeddings(text, img_path="") -> np.array:
    query = subquery_divide_query(text)
    subqueries = get_llm_response("", query).replace("\n","").split(";")
    embeddings = get_batch_embedding([EmbeddingRequestData(subquery, img_path) for subquery in subqueries])
    return np.stack(embeddings)


def subquery_divide_query(query: str) -> str:
    return f"""Instruction: You are a retrieval-oriented query decomposer.

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

def subquery_modality_query(subquery: str) -> str:
    return f"""Instruction: You are a modality selector for multimodal QA.

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

