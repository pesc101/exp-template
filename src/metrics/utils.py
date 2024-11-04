import json
import re
import string
from dataclasses import dataclass
from typing import List, TypedDict

from encourage.llm.response import Response


@dataclass
class Metric:
    score: float
    raw: List[float]
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class Context(TypedDict):
    url: str
    content: str
    score: float


class Response(TypedDict):
    id: str  # request_id
    question: str  # user_prompt
    reference_answer: str  # meta_data
    sources: List[str]  # meta_data
    contexts: List[Context]  # context
    generated_answer: str  # response
    latency: float  # processing_time


def clean_text(text: str) -> str:
    markdown_symbols = r"[\*\_\`\~\#\>\-\[\]]"
    text = re.sub(markdown_symbols, "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    text = text.strip()
    return text


def split_into_sentences(nlp, text: str) -> List[str]:
    return [sentence.text for sentence in nlp(text).sents]


def remove_punctuation(sentence):
    translator = str.maketrans("", "", string.punctuation)
    return sentence.translate(translator)


def load_run(json_path) -> List[Response]:
    with open(json_path) as fin:
        responses = json.load(fin)

    filtered = []
    for response in responses:
        # filter empty reference/generation
        if not response["reference_answer"]:
            print(f'WARNING: {response["id"]} has no reference answer. Skip.')
            continue
        if not response["generated_answer"]:
            print(f'WARNING: {response["id"]} has no generated answer. Skip.')
            continue

        filtered.append(response)

    return filtered
