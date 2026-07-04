from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()


def exercise_first_chain(product, audience):
    """
    Excercise: Create a chain that:
    1. Takes a product name and target audience
    2. Generate a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """

    prompt = ChatPromptTemplate.from_template(
        "Generate a marketing tagline given this product: {product} and target audience: {audience}. Answer in one line ONLY, do not use **"
    )

    # use this one for initializing models!
    model = init_chat_model(
        "google/gemini-2.5-flash", model_provider="openai", temperature=0.8
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    response = chain.invoke({"product": product, "audience": audience})

    print(f"Tagline for: {product} and target audience {audience}")
    print(response)
    print()


if __name__ == "__main__":
    exercise_first_chain(product="AI Course", audience="cats")
    exercise_first_chain(product="AI Course", audience="developers")
    exercise_first_chain(product="AI Course", audience="finances professionals")
    exercise_first_chain(product="AI Course", audience="chemical engineers")
