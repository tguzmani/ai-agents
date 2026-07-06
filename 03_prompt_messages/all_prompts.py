from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatMessage,
    ToolMessage,
)

from util.models import build_model, GPT_4_1_MINI

model = build_model(GPT_4_1_MINI)


def demo_basic_templates():
    """Basic ChatPromptTemplate usage"""

    simple = ChatPromptTemplate.from_template("Translate '{text}' to {language}")

    messages = simple.format_messages(text="Hello World!", language="Japanese Romaji")
    print("Simple template")
    print(f" {messages}")

    multi = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator, be concise"),
            ("human", "Translate '{text}' to {language}"),
        ]
    )

    messages = multi.format_messages(text="Good Morning", language="Romaji")
    print("\nMulti Message Template:")
    for message in messages:
        print(f"\t{type(message).__name__}: {message.content}")


def demo_message_types():
    """Working with different message types"""

    messages = [
        SystemMessage(content="You are a math tutor, be brief"),
        HumanMessage(content="What is 5 * 25?"),
        AIMessage(content="125"),
        HumanMessage(content="And if I add 25?"),
    ]

    response = model.invoke(messages)

    print(f"Conversation Result:\n{response.content}")


def demo_messages_placeholder():
    """Use MessagePlaceholder for dynamic conversations history"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a depressed human being"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    history = [
        HumanMessage(content="My name is Tomas"),
        AIMessage(content="Whatever"),
    ]

    messages = prompt.format_messages(history=history, question="What's my name?")

    print("With history placeholder:")
    for message in messages:
        print(f"\t {type(message).__name__}: {message.content[:50]}...")

    # Execute
    response = model.invoke(messages)
    print(f"\nResponse: {response.content}")


if __name__ == "__main__":
    # demo_basic_templates()
    # demo_message_types()
    demo_messages_placeholder()
