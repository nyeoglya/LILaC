import os
import re
import typing as tp

import numpy as np

from common import (
    get_clean_savepath_from_url_with_custom_extension, get_query_embedding, get_llm_response
)

MODALITY_INSTRUCTION_QUERY = {
    "text": "Given a question, retrieve text passages that answer the question",
    "table": "Given a question, retrieve table-format texts that answer the question. A table can include an image within a cell.",
    "image": "Given a question, retrieve image-description (or OCR) pairs that answer the question"
}

IMAGE_OCR_QUERY = "Extract all text from the image. Maintain the original structure. If no text is detected, return an empty string ('') only. No introduction or closing remarks."
EXPLANATION_INSTRUCTION = "Provide a concise summary of this image in 2-3 sentences. Focus on the core subject, the setting, and the most striking visual element. Avoid filler words; be direct and precise."

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

DEMONSTRATION_PROMPT = """
/*
[Table]
Title: Gonzalo Higuain
Section: Career statistics | Club

Club | Season | League - Division | League - Apps | League - Goals | National Cup - Apps | National Cup - Goals | League Cup - Apps | League Cup - Goals | Continental - Apps | Continental - Goals | Other - Apps | Other - Goals | Total - Apps | Total - Goals 
 River Plate | 2004–05 | Argentine Primera División | 4 | 0 | 0 | 0 | — | — | 0 | 0 | — | — | 4 | 0
River Plate | 2005–06 | Argentine Primera División | 14 | 5 | 0 | 0 | — | — | 4 | 2 | — | — | 18 | 7
River Plate | 2006–07 | Argentine Primera División | 17 | 8 | 0 | 0 | — | — | 2 | 0 | — | — | 19 | 8
River Plate | Total | Total | 35 | 13 | 0 | 0 | — | — | 6 | 2 | 0 | 0 | 41 | 15
Real Madrid | 2006–07 | La Liga | 19 | 2 | 2 | 0 | — | — | 2 | 0 | — | — | 23 | 2
Real Madrid | 2007–08 | La Liga | 25 | 8 | 4 | 1 | — | — | 5 | 0 | — | — | 34 | 9
Real Madrid | 2008–09 | La Liga | 34 | 22 | 2 | 1 | — | — | 7 | 0 | 1 | 1 | 44 | 24
Real Madrid | 2009–10 | La Liga | 32 | 27 | 1 | 0 | — | — | 7 | 2 | — | — | 40 | 29
Real Madrid | 2010–11 | La Liga | 17 | 10 | 2 | 1 | — | — | 6 | 2 | — | — | 25 | 13
Real Madrid | 2011–12 | La Liga | 35 | 22 | 5 | 1 | — | — | 12 | 3 | 2 | 0 | 54 | 26
Real Madrid | 2012–13 | La Liga | 28 | 16 | 5 | 0 | — | — | 9 | 1 | 2 | 1 | 44 | 18
Real Madrid | Total | Total | 190 | 107 | 21 | 4 | — | — | 48 | 8 | 5 | 2 | 264 | 121
Napoli | 2013–14 | Serie A | 32 | 17 | 5 | 2 | — | — | 9 | 5 | — | — | 46 | 24
Napoli | 2014–15 | Serie A | 37 | 18 | 4 | 1 | — | — | 16 | 8 | 1 | 2 | 58 | 29
Napoli | 2015–16 | Serie A | 35 | 36 | 2 | 0 | — | — | 5 | 2 | — | — | 42 | 38
Napoli | Total | Total | 104 | 71 | 11 | 3 | — | — | 30 | 15 | 1 | 2 | 146 | 91
Juventus | 2016–17 | Serie A | 38 | 24 | 4 | 3 | — | — | 12 | 5 | 1 | 0 | 55 | 32
Juventus | 2017–18 | Serie A | 35 | 16 | 4 | 2 | — | — | 10 | 5 | 1 | 0 | 50 | 23
Juventus | 2019–20 | Serie A | 15 | 4 | 0 | 0 | — | — | 5 | 2 | 1 | 0 | 21 | 6
Juventus | Total | Total | 88 | 44 | 8 | 5 | — | — | 27 | 12 | 3 | 0 | 126 | 61
Milan (loan) | 2018–19 | Serie A | 15 | 6 | 1 | 0 | — | — | 5 | 2 | 1 | 0 | 22 | 8
Chelsea (loan) | 2018–19 | Premier League | 14 | 5 | 2 | 0 | 1 | 0 | 2 | 0 | — | — | 19 | 5
Career total | Career total | Career total | 446 | 246 | 43 | 12 | 1 | 0 | 117 | 40 | 10 | 4 | 618 | 301
*/

/*
[Passage]
Title: 2006 FIFA Club World Cup Final

The match pitted Internacional of Brazil, the CONMEBOL club champions, against Barcelona of Spain, the UEFA club champions. Internacional won 1–0, after a counter-attack led by Iarley and the goal scored by Adriano Gabiru at the 82nd minute, in a match watched by 67,128 people. In doing so, Internacional won their first FIFA Club World Cup/Intercontinental Cup and Barcelona remained without any world club title. Deco was named as man of the match.
*/

/*
[Passage]
Title: 2018 UEFA Champions League Final

The 2018 UEFA Champions League Final was the final match of the 2017–18 UEFA Champions League, the 63rd season of Europe's premier club football tournament organised by UEFA, and the 26th season since it was renamed from the European Cup to the UEFA Champions League. It was played at the NSC Olimpiyskiy Stadium in Kiev, Ukraine on 26 May 2018, between Spanish side and defending champions Real Madrid, who had won the competition in each of the last two seasons, and English side Liverpool.
*/

Question = What club that Gonzalo Higuain played for in 2006-07 is in the champions league final?
Explanation = We first have to locate what clubs Gonzalo Higuain played in in 2006-07. We can check that there is a table of Gonzalo Higuain's career statistics. In 2006-07, he played for River Plate and Real Madrid. We then have to check which of these clubs played in the Champions League final. We can find that Real Madrid played in the 2018 UEFA Champions League Final. Therefore, the answer is: f_answers(["Real Madrid"])
                                                                                                                                                                                                                                 


/*
[Passage]
Title: South Asia

The current territories of Afghanistan, Bangladesh, Bhutan, Maldives, Nepal, India, Pakistan, and Sri Lanka form South Asia. The South Asian Association for Regional Cooperation (SAARC) is an economic cooperation organisation in the region which was established in 1985 and includes all eight nations comprising South Asia.
*/

/*
[Passage]
Title: UK Joint Expeditionary Force

The UK Joint Expeditionary Force (JEF) is a United Kingdom-led expeditionary force which may consist of, as necessary, Denmark, Finland, Estonia, Latvia, Lithuania, the Netherlands, Sweden and Norway. It is distinct from the similarly named Franco-British Combined Joint Expeditionary Force.
*/

/*
[Table]
Title: Lithuanian Armed Forces
Section: Current operations

Deployment | Organization | Operation | Personnel 
 Somalia | EU | Operation Atalanta | 15
Mali | EU | EUTM Mali | 2
Afghanistan | NATO | Operation Resolute Support | 29
Libya | EU | EU Navfor Med | 3
Mali | UN | MINUSMA | 39
Iraq | CJTF | Operation Inherent Resolve | 6
Central African Republic | EU | EUFOR RCA | 1
Kosovo | NATO | KFOR | 1
Ukraine |  | Training mission | 40
*/

Question = Among the Lithuanian Armed Forces' current operations, which of her deployments involves fewer personnel: Kosovo, or the deployment in the nation that, along with six others, constitutes the subcontinent of South Asia?
Explanation = The South Asia passage identifies Afghanistan as part of the subcontinent. The Lithuanian Armed Forces' operations table shows 29 personnel deployed in Afghanistan. In contrast, only 1 personnel is stationed in Kosovo. 1 is fewer than 29. Therefore, the answer is: f_answers(["Kosovo"]).



"""

INSTRUCTION_PROMPT_WITHOUT_IMAGE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using the f_answers() API, you can return a list of answers to a question that can be answered using the retrieved components of webpages.
A retrieved component can be a passage or a table.
Strictly follow the format of the below example. 
Return a SHORT answer to the question using the given evidences, using f_answers() API.
* For yes/no questions, only answer f_answers(["yes"]) of f_answers(["no"]).

"""
INSTRUCTION_PROMPT_WITH_IMAGE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using the f_answers() API, you can return a list of answers to a question that can be answered using the retrieved components of webpages.
A retrieved component can be a passage, a table, or an image.
Strictly follow the format of the below example.
Return a SHORT answer to the question using the given evidences, using f_answers() API.
* For yes/no questions, only answer f_answers(["yes"]) of f_answers(["no"]).

"""

def get_subembeddings(server_url: str, query_text: str, image_filepath: str = "") -> np.ndarray:
    query = subquery_divide_query(query_text)
    subqueries = get_llm_response(server_url, query).replace("\n","").split(";")
    embeddings = []
    for subquery in subqueries:
        modality = get_llm_response(server_url, subquery_modality_query(subquery))
        result_embedding = get_query_embedding(modality, subquery, image_filepath)
        embeddings.append(result_embedding)
    return np.stack(embeddings)

def subquery_divide_query(query: str) -> str:
    return SUBQUERY_DIVIDE_QUERY.format(query=query)

def subquery_modality_query(subquery: str) -> str:
    return SUBQUERY_MODALITY_QUERY.format(subquery=subquery)

class LLMQueryGenerator:
    def __init__(self, image_folderpath: str) -> None:
        self.image_folderpath: str = image_folderpath
        self.image_counter: int = 0
        self.image_link_pattern = r"\[\[([^\]]+)\]\]"

    def llm_question_query(self, question: str, component_list: tp.List[tp.Dict]) -> tp.Tuple[str, tp.List[str]]:
        self.image_counter = 0
        serialized_component_list = []
        image_filepath_list = []
        instruction_prompt = INSTRUCTION_PROMPT_WITHOUT_IMAGE
        for component_data in component_list:
            if component_data["type"] == "paragraph":
                serialized_component_list.append(self._serialize_text_component_for_prompt(component_data))
            elif component_data["type"] == "table":
                serialized_text, first_image_path = self._serialize_table_component_for_prompt(component_data)
                serialized_component_list.append(serialized_text)
                if first_image_path:
                    image_filepath_list.append(get_clean_savepath_from_url_with_custom_extension(self.image_folderpath, first_image_path, "png"))
            elif component_data["type"] == "image":
                instruction_prompt = INSTRUCTION_PROMPT_WITH_IMAGE
                serialized_text, image_path = self._serialize_image_component_for_prompt(component_data)
                serialized_component_list.append(serialized_text)
                if image_path:
                    image_filepath_list.append(get_clean_savepath_from_url_with_custom_extension(self.image_folderpath, image_path, "png"))
        
        final_text_prompt = (
            instruction_prompt
            + DEMONSTRATION_PROMPT
            + "\n"
            + "\n".join(serialized_component_list)
            + "\n"
            + f"Question = {question}\nThe answer is: "
        )
        
        return final_text_prompt, image_filepath_list

    def _serialize_text_component_for_prompt(self, text_component_data: tp.Dict) -> str:
        return f"/*\n[Passage]\nTitle: {text_component_data['doc_title']}\nSection: {', '.join(text_component_data['heading_path'])}\n\n{text_component_data['paragraph']}\n*/\n\n"

    def _serialize_table_cell_for_prompt(self, cell_textdata: str, first_image_taken: bool) -> tp.Tuple[str, str]:
        serialized_text = ""
        image_list = [item for item in re.findall(self.image_link_pattern, cell_textdata)]
        cleaned_text = re.sub(self.image_link_pattern, '', cell_textdata).strip()
        image_path = image_list[0] if image_list and not first_image_taken and os.path.exists(image_list[0]) else ""
        if image_path:
            serialized_text += f"<Image {self.image_counter}>"
            self.image_counter += 1
        if serialized_text:
            serialized_text += ", "
        serialized_text += cleaned_text
        
        normalized_serialized_text = re.sub(r'\s*,\s*', ' , ', serialized_text)
        return normalized_serialized_text, image_path

    def _serialize_table_component_for_prompt(self, table_component_data: tp.Dict) -> tp.Tuple[str, str]:
        serialized_text = f"/*\n[Table]\nTitle: {table_component_data['doc_title']}\nSection: {', '.join(table_component_data['heading_path'])}\n\n"
        first_img_taken = False
        first_image_path = ""
        
        serialized_table = []
        for table_row in table_component_data["table"]:
            serialized_cell_text_list: tp.List[str] = []
            for table_cell in table_row:
                serialized_cell_text, image_path = self._serialize_table_cell_for_prompt(table_cell, first_img_taken)
                if image_path:
                    first_img_taken = True
                    first_image_path = image_path
                serialized_cell_text_list.append(serialized_cell_text)
            serialized_table.append(" | ".join(serialized_cell_text_list))
        serialized_table.append("*/\n")

        serialized_text += '\n'.join(serialized_table) + "\n"
        return serialized_text, first_image_path

    def _serialize_image_component_for_prompt(self, image_component_data: tp.Dict) -> tp.Tuple[str, str]:
        serialized_text = f"/*\n[Image]\nTitle: {image_component_data['doc_title']}\nSection: {', '.join(image_component_data['heading_path'])}\n\n"

        image_path = get_clean_savepath_from_url_with_custom_extension(self.image_folderpath, image_component_data["src"], "png")
        if os.path.exists(image_path):
            serialized_text += f"<Image {self.image_counter}>\nCaption: {image_component_data['caption']}\n"
            self.image_counter += 1
        else:
            image_path = ""
        
        serialized_text += "*/\n\n"
        return serialized_text, image_path
