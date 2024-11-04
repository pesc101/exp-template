from typing import List

import evaluate
import ir_measures
import numpy as np
from encourage.llm.response import Response
from nltk import word_tokenize

from metrics.utils import Metric


class GeneratedAnswerLength:
    """Computes the average length of the generated answers."""

    @property
    def name(self):
        """Returns the name of the metric."""
        return "generated_answer_length"

    def __call__(self, responses: List[Response]) -> Metric:
        """Computes the average length of the generated answers."""
        lengths = [len(word_tokenize(r.response)) for r in responses]
        score = np.mean(lengths)
        return Metric(score=score, raw=lengths)


class ReferenceAnswerLength:
    """Computes the average length of the reference answers."""

    @property
    def name(self):
        """Returns the name of the metric."""
        return "reference_answer_length"

    def __call__(self, responses: List[Response]) -> Metric:
        """Computes the average length of the reference answers."""
        lengths = [len(word_tokenize(r.response)) for r in responses]
        score = np.mean(lengths)
        return Metric(score=score, raw=lengths)


class ContextLength:
    """Computes the average length of the context."""

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return "context_length"

    def __call__(self, responses: List[Response]) -> Metric:
        """Computes the average length of the context."""
        lengths = []

        for response in responses:
            context_length = 0
            for context in response["contexts"]:
                context_length += len(word_tokenize(context["content"]))
            lengths.append(context_length)

        score = np.mean(lengths)
        return Metric(score=score, raw=lengths)


class BLEU:
    """Computes the BLEU score for the generated answers."""

    def __init__(self):
        self.metric = evaluate.load("sacrebleu")

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return "bleu"

    def __call__(self, responses: List[Response]) -> Metric:
        """Computes the BLEU score for the generated answers."""
        scores = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.response for r in responses],
        )
        scores["score"] = scores["score"] / 100  # div by 100 to match with other metrics
        return Metric(score=scores["score"], raw=scores)


class ROUGE:
    """Computes the ROUGE score for the generated answers."""

    def __init__(self, rouge_type: str):
        assert rouge_type in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.metric = evaluate.load("rouge")
        self.rouge_type = rouge_type

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self.rouge_type

    def __call__(self, responses: List[Response]) -> Metric:
        """Computes the ROUGE score for the generated answers."""
        scores = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.meta_data for r in responses],
            rouge_types=[self.rouge_type],
            use_aggregator=False,
        )[self.rouge_type]
        agg = np.mean(scores)
        return Metric(score=agg, raw=scores)


class BERTScore:
    """Computes the BERTScore for the generated answers."""

    def __init__(self, **metric_args):
        self.metric = evaluate.load("bertscore")
        self.metric_args = metric_args

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return "bertscore"

    def __call__(self, responses: List[Response]) -> Metric:
        """Computes the BERTScore for the generated answers."""
        result = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.response for r in responses],
            **self.metric_args,
        )
        score = np.mean(result["f1"])
        return Metric(
            score=score,
            raw=result["f1"],
            precision=result["precision"],
            recall=result["recall"],
            f1=result["f1"],
        )


class MeanReciprocalRank:
    """Computes the Mean Reciprocal Rank (MRR) for the responses."""

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return "mrr"

    def run_to_trec(self, responses: List[Response]) -> tuple:
        """Converts responses into TREC format.

        Example:
        -------
        qrels = {
            "Q0": {"D0": 0, "D1": 1},
            "Q1": {"D0": 0, "D3": 2}
        }
        run = {
            "Q0": {"D0": 1.2, "D1": 1.0},
            "Q1": {"D0": 2.4, "D3": 3.6}
        }

        """
        qrels, run = {}, {}

        for response in responses:
            query_id = response.request_id
            relevant = {source: 1 for source in response["sources"]}
            retrieved = {context["url"]: context["score"] for context in response.context}
            qrels[query_id] = relevant
            run[query_id] = retrieved

        return qrels, run

    def __call__(self, responses: List[Response]) -> Metric:
        """Computes the Mean Reciprocal Rank (MRR) for the responses."""
        qrels, run = self.run_to_trec(responses)
        mrr = ir_measures.MRR()
        scores = [score.value for score in mrr.iter_calc(qrels, run)]
        return Metric(score=np.mean(scores), raw=scores)
