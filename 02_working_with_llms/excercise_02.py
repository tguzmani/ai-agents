from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    id: str
    price_in: float
    price_out: float

    def __str__(self) -> str:
        return self.id


MODELS: dict[str, Model] = {
    "gemini-flash": Model("google/gemini-2.5-flash", 0.30, 2.50),
    "claude-3-haiku": Model("anthropic/claude-3-haiku", 0.25, 1.25),
    "gpt-4.1-mini": Model("openai/gpt-4.1-mini", 0.40, 1.60),
}

GEMINI_FLASH_2_5 = MODELS["gemini-flash"].id
CLAUDE_3_HAIKU = MODELS["claude-3-haiku"].id
GPT_4_1_MINI = MODELS["gpt-4.1-mini"].id


def build_model(model) -> BaseChatModel:
    return init_chat_model(
        model,
        model_provider="openai",
        temperature=0.9,
        max_retries=3,
        streaming=True,
    )


def exercise_multi_model():
    """
    1. Takes a question and a list of models
    2. Gets responses from all models
    3. Returns a dict of {model_name: response}
    """

    question = input("Ask a question: ")

    models = {
        "gemini-2.5-flash": build_model(GEMINI_FLASH_2_5),
        "gpt-4.1-mini": build_model(GPT_4_1_MINI),
        "haiku-3": build_model(CLAUDE_3_HAIKU),
    }

    messages = [
        SystemMessage(content="Answer in just one sentence, do not use *"),
        HumanMessage(content=question),
    ]

    responses = {
        "gemini-2.5-flash": models["gemini-2.5-flash"].invoke(messages).content,
        "gpt-4.1-mini": models["gpt-4.1-mini"].invoke(messages).content,
        "haiku-3": models["haiku-3"].invoke(messages).content,
    }

    print(responses)


if __name__ == "__main__":
    exercise_multi_model()
