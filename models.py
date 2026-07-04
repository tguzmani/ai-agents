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
