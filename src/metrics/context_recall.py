from typing import List

import numpy as np

from metrics import context_recall_prompt
from metrics.llm import LLM
from metrics.utils import Response


class ContextRecall:
    """How complete the context is for generating the ground-truth."""

    def __init__(self, model: LLM):
        self.temperature = 0
        self.max_tokens = 4096
        self.model = model

    @property
    def name(self):
        return "context_recall"

    def format_contexts(self, contexts):
        # concatenates all contexts and escapes special characters including newlines
        text = "\n".join([context["content"] for context in contexts])
        return repr(text)

    def __call__(self, responses: List[Response]):
        # Step 1: Prompts preparation
        prompts = [
            context_recall_prompt.format_prompt(
                context_recall_prompt.Example(
                    question=response["question"],
                    context=self.format_contexts(response["contexts"]),
                    answer=response["reference_answer"],
                )
            )
            for response in responses
        ]

        # Step 2: Statements classification
        classification_results = self.model.generate_guided(
            prompts=prompts,
            schema=context_recall_prompt.ClassifiedSentencesList,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Step 3: Recall computation
        all_sentences = [result.sentences for result in classification_results]
        total = [len(result.sentences) for result in classification_results]
        attributed = [
            sum(sent.label == 1 for sent in result.sentences) for result in classification_results
        ]
        scores = [a / t if t > 0 else np.nan for a, t in zip(attributed, total)]
        agg = sum(attributed) / sum(total)

        return {
            "score": agg,
            "raw": scores,
            "total": total,
            "attributed": attributed,
            "sentences": all_sentences,
        }
