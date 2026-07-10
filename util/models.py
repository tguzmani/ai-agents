from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel

load_dotenv()


def build_model(model: str, temperature: float = 0.9) -> BaseChatModel:
    return init_chat_model(
        model,
        model_provider="openai",
        temperature=temperature,
        max_retries=3,
    )


@dataclass(frozen=True)
class Model:
    id: str
    price_in: float
    price_out: float

    def __str__(self) -> str:
        return self.id


MODELS: dict[str, Model] = {
    "gemini-flash": Model("google/gemini-2.5-flash", 0.30, 2.50),
    "deepseek_v3": Model("deepseek/deepseek-chat-v3-0324", 0.24, 0.90),
    "claude-3-haiku": Model("anthropic/claude-3-haiku", 0.25, 1.25),
    "gpt-4.1-mini": Model("openai/gpt-4.1-mini", 0.40, 1.60),
    "text-embedding-3-small": Model("openai/text-embedding-3-small", 0.02, 0.0),
}

GEMINI_FLASH_2_5 = MODELS["gemini-flash"].id
CLAUDE_3_HAIKU = MODELS["claude-3-haiku"].id
GPT_4_1_MINI = MODELS["gpt-4.1-mini"].id
DEEPSEEK_V3 = MODELS["deepseek_v3"].id
TEXT_EMBEDDING_3_SMALL = MODELS["text-embedding-3-small"].id
