from util.models import build_model, GPT_4_1_MINI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch

model = build_model(GPT_4_1_MINI)

parser = StrOutputParser()


def chat_from_template(text):
    return ChatPromptTemplate.from_template(text)


def chain_branching():
    code_prompt = chat_from_template("You are a coding expert. Help with: {input}.")

    general_prompt = chat_from_template("You are a helpful assistant. Answer: {input}")

    classifier_prompt = chat_from_template(
        "Classify this as `code` or `general`: {input}\n"
        "Return only the classificator"
    )

    classifier = classifier_prompt | model | parser

    def is_code_question(input_dict):
        classification = classifier.invoke(input_dict)
        is_code = "code" in classification.lower()

        return is_code

    branch = RunnableBranch(
        (is_code_question, code_prompt | model | parser),
        general_prompt | model | parser,
    )

    questions = [
        "How do I write a for loop in Python?",
        "What's the weather like today?",
    ]

    for question in questions:
        result = branch.invoke({"input": question})
        print(f"Question: {question}")
        print(f"Answer: {result[:100]}...\n")


if __name__ == "__main__":
    chain_branching()
