from typing import List

import nltk
import numpy as np

from metrics import answer_faithfulness_nli, answer_faithfulness_split
from metrics.llm import LLM
from metrics.utils import Response


class AnswerFaithfulness:
    def __init__(self, model: LLM):
        self.temperature = 0
        self.max_tokens = 8192
        self.model = model

    @property
    def name(self):
        return "answer_faithfulness"

    def __call__(self, responses: List[Response]):
        # Step 1: Split records into claims
        claim_prompts, response_indices = [], []
        for response_idx, response in enumerate(responses):
            sentences = nltk.sent_tokenize(response["generated_answer"])
            for sent in sentences:
                prompt = answer_faithfulness_split.format_prompt(
                    answer_faithfulness_split.Example(
                        question=response["question"],
                        answer=response["generated_answer"],
                        sentence=sent,
                    )
                )
                claim_prompts.append(prompt)
                response_indices.append(response_idx)

        # Step 2: Generate claims
        claims = self.model.generate_guided(
            prompts=claim_prompts,
            schema=answer_faithfulness_split.Output,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Step 3: Gather claims per record
        response_to_claims = [[] for _ in range(len(responses))]
        for response, response_idx in zip(claims, response_indices):
            response_to_claims[response_idx] += response.simpler_statements

        # Step 4: Prepare NLI prompts
        nli_prompts = []
        for response_idx, claims in enumerate(response_to_claims):
            response = responses[response_idx]
            x = answer_faithfulness_nli.Example(
                context=" ".join([context["content"] for context in response["contexts"]]),
                statements=claims,
            )
            prompt = answer_faithfulness_nli.format_prompt(x)
            nli_prompts.append(prompt)

        # Step 5: Perform NLI
        # FIX: when claims is empty, it's not clear what this generation will return.
        nli_responses = self.model.generate_guided(
            nli_prompts,
            schema=answer_faithfulness_nli.Output,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Step 6: Process results
        supported = [sum(v.verdict == 1 for v in res.verdicts) for res in nli_responses]
        total = [len(res.verdicts) for res in nli_responses]
        scores = [s / t if t > 0 else np.nan for s, t in zip(supported, total)]
        claims = [res.verdicts for res in nli_responses]

        # micro-average over all responses
        agg = sum(supported) / sum(total)

        return {
            "score": agg,
            "raw": scores,
            "supported": supported,
            "total": total,
            "claims": claims,
        }
