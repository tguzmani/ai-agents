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

# Few Shots

model = build_model(GEMINI_FLASH_2_5)

examples = [{"input": "happy", "output": "sad"}, {"input": "tall", "output": "short"}]

example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("ai", "{output}")]
)

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt, examples=examples
)

# print(f"Fewshot Promps: {fewshot_prompt}")

final_prompt = ChatPromptTemplate.from_messages(
    [("system", "Answer with just one word only"), fewshot_prompt, ("human", "{input}")]
)

response = model.invoke(final_prompt.format_messages(input="hurr"))

print(f"Few Shot Response:\n {response.content}")
