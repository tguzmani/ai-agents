from util.models import build_model, GPT_4_1_MINI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

model = build_model(GPT_4_1_MINI)

text = {
    "text": "World War II, or the Second World War (1 September 1939 – 2 September 1945), was a global conflict between two coalitions: the Allies and the Axis powers. Nearly all of the world's countries participated. World War II was the deadliest conflict in history, causing the death of 60 to 75 million people. Millions died as a result of massacres, starvation, disease, and genocides, including the Holocaust. After the Allied victory, Germany, Austria, Japan, and Korea were occupied, and German and Japanese leaders were tried for war crimes."
}

parser = StrOutputParser()


def basic_chain():
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {text}"
    )

    chain = prompt | model | parser

    result = chain.invoke(text)

    print(f"Summary: {result}")


def parallel_chain():
    """Run multiple chains in parallel"""

    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize in two setences: {text}"
    )

    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract the keywords in the following text: {text}. Return as comma separated list"
    )

    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the following text? {text}"
    )

    model_parser_chain = model | parser

    # Parallel execution
    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model_parser_chain,
        keyword=keywords_prompt | model_parser_chain,
        sentiment_prompt=sentiment_prompt | model_parser_chain,
    )

    result = analysis_chain.invoke(text)

    for key, value in result.items():
        print(f"Key: {key}\nValue:{value}\n\n")


if __name__ == "__main__":
    # basic_chain()
    parallel_chain()
