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


def multi_conversation_memory():
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

    user_a_config = {"configurable": {"session_id": "user_a"}}
    user_b_config = {"configurable": {"session_id": "user_b"}}

    # User A conversation
    print(f"User A Conversations")
    print(f"\nUser A: Hi! My name is Tomas and my favorite language is Python")
    response = chain_with_history.invoke(
        {"input": "Hi! My name is Tomas and my favorite language is Python"},
        config=user_a_config,
    )
    print(f"AI: {response}")

    # User B conversation
    print(f"\nUser B Conversations")
    print(f"\nUser B: Hi! My name is Alice and my favorite language is JavaScript")
    response = chain_with_history.invoke(
        {"input": "Hi! My name is Alice and my favorite language is JavaScript"},
        config=user_b_config,
    )
    print(f"AI: {response}")

    # Ask the model about their favorite languages
    print(f"\nUser A: What is my favorite programming language?")
    response = chain_with_history.invoke(
        {"input": "What is my favorite programming language?"}, config=user_a_config
    )

    print(f"AI: {response}")

    # Now user B
    print(f"\nUser B: What is my favorite programming language?")
    response = chain_with_history.invoke(
        {"input": "What is my favorite programming language?"}, config=user_b_config
    )
    print(f"AI: {response}")


if __name__ == "__main__":
    multi_conversation_memory()
