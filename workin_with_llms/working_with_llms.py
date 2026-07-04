"""
Working with LLMs in LangChain V1
Multi provider, config, streaming and cost optimization
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

GEMINI_FLASH = "google/gemini-2.5-flash"
COHERE_7B = "cohere/command-r7b-12-2024"


def build_model(model):
    return init_chat_model(
        model,
        model_provider="openai",
        temperature=0.9,
        max_retries=3,
        streaming=True,
    )


def demo_model_comparison(prompt):
    models = {
        "gemini-flash": build_model(GEMINI_FLASH),
        "command-7b": build_model(COHERE_7B),
    }

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from: {model_name}: {response.content}\n")


def demo_message():
    model = build_model(GEMINI_FLASH)

    messages = [
        SystemMessage(
            content="You are a greedy assistant. Answer in one sad sentence, do not use *"
        ),
        HumanMessage(content="Explain me why the sky is blue"),
    ]

    response = model.invoke(messages)

    print(f"Response using message objects: {response.content}")


if __name__ == "__main__":
    # Model comparison

    # prompt = "Carwash is 10 meters away from home. Should I go in my car or by foot? One sentence"
    # demo_model_comparison(prompt)

    demo_message()
