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


class WindowedChatHistory(InMemoryChatMessageHistory):
    """Chat history that keeps only last k messages pairs"""

    k: int = 3

    def add_messages(self, messages):
        super().add_messages(messages)

        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-(self.k * 2) :]


def window_memory():
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowedChatHistory(k=2)

        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise and to the point."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | model | parser

    store: dict[str, WindowedChatHistory] = {}

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "user_a"}}

    coversation = [
        "My name is Tomas",
        "I work as an AI Engineer",
        "I have 4 cats",
        "I live in Venezuela",
        "Tell me everything you know about me",
    ]

    for i, message in enumerate(coversation):
        print(f"\nUser A: {message}")
        response = chain_with_history.invoke({"input": message}, config=config)
        print(f"AI: {response}")

        history = get_session_history("user_a").messages
        print(f"History length after message {i + 1}: {len(history)}")
        facts_in_memory = [
            messsage.content for messsage in history if messsage.type == "human"
        ]
        print(f"Facts in memory after message {i + 1}: {facts_in_memory}")


if __name__ == "__main__":
    window_memory()
