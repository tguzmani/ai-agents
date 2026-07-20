import tiktoken
from langchain.messages import trim_messages
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables import RunnableWithMessageHistory

from util.models import (
    GPT_4_1_MINI,
    TEXT_EMBEDDING_3_SMALL,
    build_embeddings,
    build_model,
)

model = build_model(GPT_4_1_MINI)
parser = StrOutputParser()


def tiktoken_counter(messages: list[BaseMessage]) -> int:
    # gpt-4.1 / gpt-4o usan la codificación o200k_base
    enc = tiktoken.get_encoding("o200k_base")
    # +3 tokens de "overhead" por mensaje (formato de rol interno de OpenAI)
    return sum(3 + len(enc.encode(str(msg.content))) for msg in messages)


def message_trimming():
    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="What is Python?"),
        AIMessage(
            content="Python is a high-level programming language known for readability and simplicity."
        ),
        HumanMessage(content="How do I install it?"),
        AIMessage(
            content="You can install Python from python.org or use package managers like apt, brew, or pyenv."
        ),
        HumanMessage(content="What about pip?"),
        AIMessage(
            content="Pip is Python's package installer. It comes with Python 3.4 and later."
        ),
        HumanMessage(content="Can you summarize everything we discussed?"),
    ]

    print(f"Orifinal messages length: {len(messages)}")

    trimmed_messages = trim_messages(
        messages,
        max_tokens=50,
        strategy="last",
        token_counter=tiktoken_counter,
        include_system=True,
        allow_partial=False,
    )

    print(f"Trimmed messages length: {len(trimmed_messages)}")

    for message in trimmed_messages:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    message_trimming()
