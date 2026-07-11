from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def build_model(model: str, temperature: float = 0.9) -> BaseChatModel:
    return init_chat_model(
        model,
        model_provider="openai",
        temperature=temperature,
        max_retries=3,
    )


def build_embeddings(model: str) -> OpenAIEmbeddings:
    # Igual que build_model: OpenAIEmbeddings hereda OPENAI_API_KEY y
    # OPENAI_API_BASE del entorno, así que sale por OpenRouter sin config extra.
    #
    # check_embedding_ctx_length=False: manda el texto crudo a la API en vez de
    # tokenizar localmente con tiktoken. tiktoken no sabe mapear el id de
    # OpenRouter ("openai/text-embedding-3-small") a un tokeniser, y su fallback
    # (descargar cl100k_base) se cuelga si esa descarga no responde. OpenRouter
    # tokeniza del lado del servidor, así que no perdemos nada.
    return OpenAIEmbeddings(model=model, max_retries=3, check_embedding_ctx_length=False)


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
