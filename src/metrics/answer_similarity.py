import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from metrics.utils import Response

logger = logging.getLogger(__name__)


class AnswerSimilarity:
    """Estimate the similarity between answer and reference embeddings."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    @property
    def name(self):
        return "answer_similarity"

    def _get_embedding(self, text: Union[List[str], str]) -> np.ndarray:
        if isinstance(text, str):
            return self.model.encode([text])
        return self.model.encode(text)

    def __call__(self, responses: List[Response]):
        generated_and_reference = [
            (response["generated_answer"], response["reference_answer"])
            for response in responses
            if response["generated_answer"] is not None
        ]
        generated, reference = zip(*generated_and_reference)
        answer_emb = self._get_embedding(list(generated))
        reference_emb = self._get_embedding(list(reference))

        similarity_matrix = self.model.similarity(answer_emb, reference_emb)
        similarities = similarity_matrix.diagonal().tolist()
        return {"score": np.mean(similarities), "raw": similarities}
