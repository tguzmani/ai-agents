from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables import RunnableWithMessageHistory

from util.models import (
    GEMINI_FLASH_2_5,
    TEXT_EMBEDDING_3_SMALL,
    build_embeddings,
    build_model,
)

model = build_model(GEMINI_FLASH_2_5)
parser = StrOutputParser()


def basic_memory():
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()

        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise and to the point."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | model | parser

    store: dict[str, InMemoryChatMessageHistory] = {}

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "user_123"}}

    messages = [
        "Hi! My name is Tomas",
        "I am learning about LangChain",
        "What is my name and what am I learning?",
    ]

    print(f"Conversations")
    for message in messages:
        print(f"\nUser: {message}")
        response = chain_with_history.invoke({"input": message}, config=config)
        print(f"AI: {response}")

    # Show stored memory with length of messages
    print(f"\nStored memory for session_id 'user_123':")

    history = get_session_history("user_123")
    history_length = len(history.messages)

    print(f"Number of messages in memory: {history_length}")
    print("Messages in memory:")
    for i, msg in enumerate(history.messages):
        role = "User" if msg.type == "human" else "AI"
        print(f"{i + 1}. {role}: {msg.content}")


if __name__ == "__main__":
    basic_memory()
