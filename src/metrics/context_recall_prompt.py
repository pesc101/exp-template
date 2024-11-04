from typing import List, Literal, Optional

import outlines
from pydantic import BaseModel


class ClassifiedSentence(BaseModel):
    sentence: str
    reason: str
    label: Literal[0, 1]


class ClassifiedSentencesList(BaseModel):
    sentences: List[ClassifiedSentence]


class Example(BaseModel):
    question: str
    context: str
    answer: str
    classification: Optional[ClassifiedSentencesList] = None


INSTRUCTION = "Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Provide the answer sentence, a rationale for your decision and the final classification label. Use only 1 (Yes) or 0 (No) as a binary classification."


EXAMPLE_1 = Example(
    question="What can you tell me about albert Albert Einstein?",
    context="Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
    answer="Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895",
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
                reason="The date of birth of Einstein is mentioned clearly in the context.",
                label=1,
            ),
            ClassifiedSentence(
                sentence="He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
                reason="The exact sentence is present in the given context.",
                label=1,
            ),
            ClassifiedSentence(
                sentence="He published 4 papers in 1905.",
                reason="There is no mention about papers he wrote in the given context.",
                label=0,
            ),
            ClassifiedSentence(
                sentence="Einstein moved to Switzerland in 1895.",
                reason="There is no supporting evidence for this in the given context.",
                label=0,
            ),
        ]
    ),
)


EXAMPLE_2 = Example(
    question="who won 2020 icc world cup?",
    context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
    answer="England",
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="England",
                reason="From context it is clear that England defeated Pakistan to win the World Cup.",
                label=1,
            )
        ]
    ),
)

EXAMPLE_3 = Example(
    question="What is the primary fuel for the Sun?",
    context="",
    answer="Hydrogen",
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="Hydrogen",
                reason="The context contains no information",
                label=0,
            )
        ]
    ),
)


@outlines.prompt
def format_prompt(
    task: Example,
    instruction=INSTRUCTION,
    output_model=ClassifiedSentencesList,
    examples=[EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
):  # pylint: disable=W0613
    """
    ## Instruction
    {{ instruction }}

    The output should be a well-formatted JSON instance that conforms to the JSON schema below.

    {{ output_model.model_json_schema() }}

    ## Examples
    {% for example in examples %}
    Question: "{{ example.question }}"
    Context: "{{ example.context }}"
    Answer: "{{ example.answer }}"
    Classification: {{ example.model_dump()['classification'] }}

    {% endfor %}

    ## Task
    Question: "{{ task.question }}"
    Context: "{{ task.context }}"
    Answer: "{{ task.answer }}"
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
