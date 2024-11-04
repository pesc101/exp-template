from typing import List

from metrics import non_answer_critic_prompt
from metrics.llm import LLM
from metrics.utils import Response


class NonAnswerCritic:
    """Check if generated_answer is a non-answer."""

    def __init__(self, model: LLM):
        self.model = model
        self.temperature = 0
        self.max_tokens = 2048

    @property
    def name(self):
        return "non-answer_critic"

    def __call__(self, questions_responses: List[Response]) -> float:
        # Step 1: Prompts preparation
        prompts = [
            non_answer_critic_prompt.format_prompt(
                non_answer_critic_prompt.Example(
                    answer=response["generated_answer"],
                )
            )
            for response in questions_responses
            if response["generated_answer"] is not None
        ]

        # Step 2: Context verification
        answer_results = self.model.generate_guided(
            prompts=prompts,
            schema=non_answer_critic_prompt.ClassifiedAnswer,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Step 3: Computation
        good_answers = [not answer.non_answer for answer in answer_results]

        return {
            "score": sum(good_answers) / len(answer_results),
            "raw": [answer.non_answer for answer in answer_results],
            "rationales": [answer.rationale for answer in answer_results],
        }
