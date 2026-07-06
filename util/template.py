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

from util.models import build_model, DEEPSEEK_V3

model = build_model(DEEPSEEK_V3)
