from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from metrics import answer_relevance_qgen
from metrics.llm import LLM
from metrics.non_answer_critic import NonAnswerCritic
from metrics.utils import Response


class AnswerRelevance:
    """How relevant the answer is to the question."""

    def __init__(
        self,
        model: LLM,
        embeddings_model_name: str = "all-mpnet-base-v2",
        n_questions=3,
        temperature=0.7,
    ):
        self.n = n_questions
        self.temperature = temperature
        self.max_tokens = 2048
        self.model = model
        self.embeddings_model = SentenceTransformer(embeddings_model_name)
        self.non_answer_critic = NonAnswerCritic(model=model)

    @property
    def name(self):
        return "answer_relevance"

    def question_similarity(self, question: str, generated: List[str]):
        q_embedding = self.embeddings_model.encode([question])
        gen_embeddings = self.embeddings_model.encode(generated)
        similarities = self.embeddings_model.similarity(q_embedding, gen_embeddings)[0]
        return similarities.mean().item()

    def format_contexts(self, contexts):
        # concatenates all contexts and escapes special characters including newlines
        text = "\n".join([context["content"] for context in contexts])
        return repr(text)

    def __call__(self, responses: List[Response]):
        # Step 1: identify noncomittal answers
        result = self.non_answer_critic(responses)
        noncomittal = result["raw"]
        rationales = result["rationales"]

        # 0 = answer
        # 1 = non-answer
        committal_responses = [
            response for response, label in zip(responses, noncomittal) if label == 0
        ]
        committal_ixs = [i for i, label in enumerate(noncomittal) if label == 0]

        # Step 2: generate questions (only for valid answers)
        prompts = [
            answer_relevance_qgen.format_prompt(
                answer_relevance_qgen.Example(
                    context=self.format_contexts(response["contexts"]),
                    answer=response["generated_answer"],
                )
            )
            for response in committal_responses
        ]
        generated_questions = []
        for _ in range(self.n):
            gens = self.model.generate_guided(
                prompts=prompts,
                schema=answer_relevance_qgen.GeneratedQuestion,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            generated_questions.append([gen.question for gen in gens])
        # reshape from (n, len(prompts)) to (len(prompts), n)
        generated_questions = [list(questions) for questions in zip(*generated_questions)]

        # Step 3: Relevance calculation
        scores = [
            self.question_similarity(
                response["question"],
                generated,
            )
            for response, generated in zip(committal_responses, generated_questions)
        ]

        # Return scores for all responses, where non-answers have a None score.
        full_scores = [None] * len(responses)
        for i, score in zip(committal_ixs, scores):
            full_scores[i] = score

        return {
            "score": np.mean(scores),
            "raw": full_scores,
            "noncommittal": noncomittal,
            "rationales": rationales,
            "generated_questions": generated_questions,
        }
