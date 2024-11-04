from typing import Literal, Optional

import outlines
from pydantic import BaseModel


class ClassifiedAnswer(BaseModel):
    rationale: str
    non_answer: Literal[0, 1]


class Example(BaseModel):
    answer: str
    classification: Optional[ClassifiedAnswer] = None


INSTRUCTION = """
Classify the given response as either an answer or a non-answer.

Non-answers are responses that indicate a lack of information or inability to provide an answer. They typically include phrases such as:
- "I don't know..."
- "I can't provide information about..."
- "It is not stated..."
- "There is no answer"
- "There is no information..."
- Or other synonymous expressions indicating uncertainty or lack of information

Any response that provides concrete information or a direct answer, even if brief, should be classified as an answer.
"""
EXAMPLE_1 = Example(
    answer="The information provided does not mention anything about compulsory elective modules or the number of free modules a student can choose from in the M.Sc. Data Science program at Marburg University.",
    classification=ClassifiedAnswer(
        rationale="The response indicates a lack of information, which is characteristic of a non-answer.",
        non_answer=1,
    ),
)

EXAMPLE_2 = Example(
    answer="The question is not clear because there is no specific module mentioned. The text only talks about enrollment as a doctoral student and provides information about the required documents and the enrollment process. It does not mention a specific module.",
    classification=ClassifiedAnswer(
        rationale="The response states that specific information is not provided, making it a non-answer.",
        non_answer=1,
    ),
)


EXAMPLE_3 = Example(
    answer="Master of Science (M.Sc.)",
    classification=ClassifiedAnswer(
        rationale="The response provides a specific piece of information, making it a valid answer.",
        non_answer=0,
    ),
)


@outlines.prompt
def format_prompt(
    task: Example,
    instruction=INSTRUCTION,
    output_model=ClassifiedAnswer,
    examples=[EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
):  # pylint: disable=W0613
    """
    ## Instruction
    {{ instruction }}

    For each response, provide a classification in the following format:
    Classification: {'rationale': 'Brief explanation of why it's an answer or non-answer', 'non_answer': 0 or 1}


    Use 0 for answers and 1 for non-answers in the 'non_answer' field.

    ## Examples
    {% for example in examples %}
    Response: {{example.answer}}
    Classification: {{ example.model_dump()['classification'] }}

    {% endfor %}

    Now, classify the following response:
    Response: {{task.answer}}
    Classification:
    """


if __name__ == "__main__":
    task = Example(
        question="What is the degree for this Data Science program?",
        context="Main Content\n\n# Perspectives\n\nAre you interested in the M.Sc. Data Science program and wondering what you can do with the degree? Or you are already close to graduation and wonder what now? Below you will find information about career prospects.\n\nAlle Elemente ausklappen Alle Elemente einklappen \n\n  * ### [Inhalt ausklappen Inhalt einklappen Qualifications][654]\n\nThe master's degree program in Data Science prepares you for future work in the field of data science in science and industry, for which in-depth knowledge of computer science as well as applied mathematics is required. The study program serves to deepen and specialize the previous bachelor's degree. In addition to the technical skills, such as designing efficient algorithms and implementing them in distributed environments, you will acquire many other qualifications during your studies that you can use in your later professional life.\n\n**Project Work**\n\nThrough the project-oriented structure of the program, you will learn to work on difficult issues both independently and scientifically as part of a team. You will be able to plan larger projects over a longer period of time and complete them on time. You will then be able to present your results in a professional manner and adapted to your target group.\n\n**Key Qualifications**\n\nIn addition, you will learn how to assess yourself: How well can you work under time pressure and organize yourself? What are your strengths, what are your weaknesses? These insights will help you and your future employers or business partners. As a student of the Data Science program, you will be guided to teamwork and independent learning in the various internships and exercises, which will increasingly promote social and communication skills.\n\n  * ### [Inhalt ausklappen Inhalt einklappen Fields of Activity][654]\n\n**Exzellent Career Prospects**\n\nIn almost all scientific disciplines as well as in industry and business, the collection of massive amounts of data results in problems when dealing with this data. Many disciplines and companies lack the knowledge and tools to deal with these data volumes and to derive their benefits from them. Well-educated graduates of the Data Science program can perform important functions here because they have been specifically trained to solve such problems - there are therefore excellent career prospects!\n\n**Big Data - everywhere!**\n\nBig data is now present in almost all areas of life, even if this is often not perceived. The fields of activity for data analysts are therefore diverse and interdisciplinary - they can be found in research institutions as well as in virtually all areas of industry and business, such as in the banking and insurance sectors, the automotive industry or innovative companies in the environment of the internet.\n\n  * ### [Inhalt ausklappen Inhalt einklappen Doctorate][654]\n\nIn your master's degree program, you have learned to work scientifically on your own. In addition, you have acquired in-depth expertise in a special field of computer science or applied mathematics. If you are interested in working on long-term issues (such as your master's thesis), you can further deepen your knowledge and skills with us through subsequent doctoral studies and work on a current research problem.\n\nIf you are interested in a doctorate, please contact the professor whose field interests you the most.",
        answer="The degree for this Data Science program is a Master's degree (M.Sc.).",
    )
    prompt = format_prompt(task)
    print(prompt)
