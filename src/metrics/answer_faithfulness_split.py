from typing import List, Optional

import outlines
from pydantic import BaseModel


class Output(BaseModel):
    simpler_statements: List[str]


class Example(BaseModel):
    question: str
    answer: str
    sentence: str
    output: Optional[Output] = None


INSTRUCTION = "You are given a question, an answer, and one sentence extracted from the answer. Please break down the sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement."

EXAMPLE_1 = Example(
    question="Who was Albert Einstein and what is he best known for?",
    answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    sentence="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time.",
    output=Output(
        simpler_statements=[
            "Albert Einstein was a German-born theoretical physicist.",
            "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
        ]
    ),
)

EXAMPLE_2 = Example(
    question="Who was Albert Einstein and what is he best known for?",
    answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    sentence="He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    output=Output(
        simpler_statements=[
            "Albert Einstein was best known for developing the theory of relativity.",
            "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
        ]
    ),
)


@outlines.prompt
def format_prompt(
    task: Example,
    instruction=INSTRUCTION,
    output_model=Output,
    examples=[EXAMPLE_1, EXAMPLE_2],
):  # pylint: disable=W0613
    """
    ## Instruction
    {{ instruction }}

    The output should be a well-formatted JSON instance that conforms to the JSON schema below.

    {{ output_model.model_json_schema() }}

    ## Examples
    {% for example in examples %}
    Question: {{ example.question }}
    Answer: {{ example.answer }}
    Sentence: {{ example.sentence }}
    Analysis: {{ example.model_dump()['output'] }}

    {% endfor %}

    ## Task
    Question: {{ task.question }}
    Answer: {{ task.answer }}
    Sentence: {{ task.sentence }}
    Analysis:
    """


if __name__ == "__main__":
    task = Example(
        question="Who is the president of the united states?",
        answer="As of July 2024, the President of the United States is Joseph R. Biden Jr. He assumed office as the 46th President on January 20, 2021, after defeating the incumbent, Donald Trump, in the 2020 presidential election.",
        sentence="As of July 2024, the President of the United States is Joseph R. Biden Jr.",
    )
    prompt = format_prompt(task)
    print(prompt)
