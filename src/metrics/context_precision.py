from typing import List

import numpy as np

from metrics import context_precision_prompt
from metrics.llm import LLM
from metrics.utils import Response


class ContextPrecision:
    """How relevant the context is to the ground-truth answer."""

    def __init__(self, model: LLM):
        self.model = model
        self.temperature = 0
        self.max_tokens = 2048

    @property
    def name(self):
        return "context_precision"

    def _average_precision(self, labels: List[int]) -> float:
        """Computes average precision over a list of ranked results. Labels should be a list of binary labels, where 1 is relevant, and 0 is irrelevant."""
        denominator = sum(labels) + 1e-10
        numerator = sum([(sum(labels[: i + 1]) / (i + 1)) * labels[i] for i in range(len(labels))])
        score = numerator / denominator
        return score

    def __call__(self, responses: List[Response]):
        # Step 1: Prompts preparation
        prompts = [
            context_precision_prompt.format_prompt(
                context_precision_prompt.Example(
                    question=response["question"],
                    context=context["content"],
                    answer=response["reference_answer"],
                )
            )
            for response in responses
            if response["contexts"] is not None
            for context in response["contexts"]
        ]

        # Step 2: Context verification
        verification_results = self.model.generate_guided(
            prompts=prompts,
            schema=context_precision_prompt.Verdict,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Step 3: Precision computation
        precisions_per_questions = []
        all_labels = []
        current_idx = 0
        for response in responses:
            contexts_cnt = len(response["contexts"])
            verdicts = verification_results[current_idx : current_idx + contexts_cnt]
            labels = [verdict.verdict for verdict in verdicts]
            all_labels.append(labels)
            precision = self._average_precision(labels)
            precisions_per_questions.append(precision)
            current_idx += contexts_cnt

        agg = np.mean(precisions_per_questions)

        # Step 4: Detailed Output
        return {
            "score": agg,
            "raw": precisions_per_questions,
            "labeled_contexts": all_labels,
        }
