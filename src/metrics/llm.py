from typing import Dict, List, Union

try:
    import vllm
    from outlines.serve.vllm import JSONLogitsProcessor
except ModuleNotFoundError:
    print("WARNING: Need to install vllm to run LLM-based metrics.")
    vllm = None
    JSONLogitsProcessor = None

from pydantic import BaseModel, ValidationError


class LLM:
    def __init__(self, model, **vllm_args):
        self.model = vllm.LLM(model, **vllm_args)
        self.tokenizer = self.model.get_tokenizer()

    def to_chat(
        self,
        inputs: Union[str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]],
    ):
        """Applies chat template to the inputs.

        Valid inputs are:
        1. A string
        2. A list of strings
        3. A conversation in form [{"role": "...", "content": "..."}, ...]
        4. A list of conversations.
        """
        if isinstance(inputs, str):
            inputs = [[{"role": "user", "content": inputs}]]
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            inputs = [[{"role": "user", "content": x}] for x in inputs]
        elif isinstance(inputs, list) and isinstance(inputs[0], dict):
            inputs = [inputs]

        return self.tokenizer.apply_chat_template(inputs, tokenize=False)

    def generate_guided(self, prompts, schema: BaseModel, max_retries=3, **sampling_params):
        prompts = self.to_chat(prompts)
        sampling_params = vllm.SamplingParams(
            **sampling_params,
            logits_processors=[JSONLogitsProcessor(schema=schema, llm=self.model.llm_engine)],
        )
        original_temp = sampling_params.temperature
        original_max_tokens = sampling_params.max_tokens

        # Pre-allocate responses, initially start with all prompts
        all_responses = [None] * len(prompts)
        failed_indices = list(range(len(prompts)))

        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            if not failed_indices:
                break  # All prompts succeeded
            elif failed_indices and attempt > 0:
                print(f"RETRY {len(failed_indices)} prompts.")

            # Increase temperature for retries:
            # If temperature is low (<= 0.7), increase it by some increment at each retry.
            # If temperature is high (> 0.7), keep it as is and just sample again.
            if original_temp <= 0.7:
                increment = 0.25
                max_temp = 0.7
                new_temp = min(original_temp + (attempt * increment), max_temp)
                sampling_params.temperature = new_temp

            # Increase max_tokens for retries
            new_max_tokens = min((attempt + 1) * original_max_tokens, 16384)
            sampling_params.max_tokens = new_max_tokens

            current_prompts = [prompts[i] for i in failed_indices]
            responses = self.model.generate(current_prompts, sampling_params)

            new_failed_indices = []
            for i, response in zip(failed_indices, responses):
                try:
                    if sampling_params.n > 1:
                        # if n > 1 we use a "all-or-nothing" policy:
                        # The prompt will only have a result if all n responses can be parsed
                        validated_response = [
                            schema.model_validate_json(output.text) for output in response.outputs
                        ]
                    else:
                        validated_response = schema.model_validate_json(response.outputs[0].text)
                    all_responses[i] = validated_response
                except ValidationError:
                    print("WARNING: failed to parse this response here")
                    print(response.outputs[0].text)
                    new_failed_indices.append(i)

            failed_indices = new_failed_indices

        return all_responses
