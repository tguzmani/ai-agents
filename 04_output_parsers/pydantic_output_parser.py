from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from util.models import build_model, GEMINI_FLASH_2_5


class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's occupation")


model = build_model(GEMINI_FLASH_2_5)

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Consider current year is 2026."
    "Return a JSON object with 'name', 'occupation' and 'age'"
    "for: {description}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

response = chain.invoke(
    {
        "description": "Tomas Guzman was born on may 23 1992"
        "and now is working as a Software Engineer in Talkive"
    }
)

print(type(response))
print(f"Response: {response}")
