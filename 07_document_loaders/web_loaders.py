from util.models import build_model, GPT_4_1_MINI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_community.document_loaders import WebBaseLoader

model = build_model(GPT_4_1_MINI)

parser = StrOutputParser()


def load_web():
    try:
        loader = WebBaseLoader(
            "https://www.bcv.org.ve/",
            bs_kwargs={"parse_only": None},
            requests_kwargs={"verify": False},
        )
        documents = loader.load()

        for document in documents:
            print(f"{document}")

    except Exception as e:
        print(f"Error loading web {e}")


load_web()
