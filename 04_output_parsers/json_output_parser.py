from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from util.models import build_model, GEMINI_FLASH_2_5

model = build_model(GEMINI_FLASH_2_5)

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Consider current year is 2026. Return a JSON object with 'first_name', 'last_name' and 'age' for: {description}"
)

chain = prompt | model | parser

response = chain.invoke({"description": "Tomas Guzman was born on may 23 1992"})

print(type(response))
print(f"Response: {response}")
