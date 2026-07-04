"""
Working with LLMs in LangChain V1
Multi provider, config, streaming and cost optimization
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from models import GEMINI_FLASH_2_5, GPT_4_1_MINI

load_dotenv()


def build_model(model) -> BaseChatModel:
    return init_chat_model(
        model,
        model_provider="openai",
        temperature=0.9,
        max_retries=3,
        streaming=True,
    )


def demo_model_comparison(prompt):
    models = {
        "gemini-flash": build_model(GEMINI_FLASH_2_5),
        "gpt-4.1-mini": build_model(GPT_4_1_MINI),
    }

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from: {model_name}: {response.content}\n")


def demo_message():
    model = build_model(GEMINI_FLASH_2_5)

    messages = [
        SystemMessage(
            content="You are a passionate assistant. "
            "Answer in one blissfull sentence, do not use *"
        ),
        HumanMessage(content="Explain me why the sky is blue"),
    ]

    response = model.invoke(messages)

    print(f"Response:\n {response.content}")

    # Multi-turn conversatoin using message objects
    messages.append(response)
    messages.append(HumanMessage(content="Why is not red?"))

    response = model.invoke(messages)

    print(f"Follow up response: {response.content}")


if __name__ == "__main__":
    # Model comparison

    # prompt = "Carwash is 10 meters away from home. Should I go in my car or by foot? One sentence"
    # demo_model_comparison(prompt)

    demo_message()
