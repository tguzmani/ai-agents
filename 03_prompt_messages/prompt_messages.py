from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatMessage,
    ToolMessage,
)

from util.models import build_model, GEMINI_FLASH_2_5

load_dotenv()

# Chat Prompt
# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
# messages = prompt.format_messages(adjective="funny", topic="whales")

# Multi message templates

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You help people translating from {input_language} to {output_language}",
        ),
        ("human", "Translate the following text: {text}"),
    ]
)

messages = prompt.format_messages(
    input_language="English",
    output_language="French",
    text="Diddy Kong bananas are red... how is this possible?",
)

model = build_model(GEMINI_FLASH_2_5)

response = model.invoke(messages)

print(f"Response: {response.content}")
