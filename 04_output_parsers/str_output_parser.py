from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from util.models import build_model, GEMINI_FLASH_2_5

model = build_model(GEMINI_FLASH_2_5)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}")

chain = prompt | model | parser

response = chain.invoke({"topic": "a lonely robot"})

print(type(response))
print(f"Response: {response}")
