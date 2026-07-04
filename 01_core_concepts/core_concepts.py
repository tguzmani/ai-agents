from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="google/gemini-2.5-flash", temperature=0.7)
parser = StrOutputParser()


def demo_basic_chain():
    """Demostrates a basic chain using LCEL and Runnables"""

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )

    # Compose with pipe operator (LCEL)

    chain = prompt | model | parser

    # Execute the chain

    result = chain.invoke({"question": "What is LangChain?"})

    print(f"Response: {result}")

    return chain


def demo_batch_chain():
    """Demostrates batch execution for multiple inputs"""

    prompt = ChatPromptTemplate.from_template("Translate to Spanish: {text}")

    chain = prompt | model | parser

    # Batch inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name"},
        {"text": "Where is the nearest restaurant"},
    ]

    results = chain.batch(inputs)

    # Now we can iterate over the results

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")


def demo_streaming():
    """Demostrate streaming for real-time output"""

    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")

    chain = prompt | model | parser

    print("Streaming output:")

    for chunk in chain.stream({"topic": "volcano"}):
        print(chunk, end="", flush=True)

    print()


def demo_schema_inspection():
    """Demostrate input/output schema inspection"""

    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")

    chain = prompt | model | parser

    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_chain()
    # demo_streaming()
    demo_schema_inspection()
