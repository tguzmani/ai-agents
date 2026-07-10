from util.models import build_model, GPT_4_1_MINI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_community.document_loaders import TextLoader

model = build_model(GPT_4_1_MINI)

parser = StrOutputParser()


def load_text_file():
    try:
        loader = TextLoader("07_document_loaders/docs/saturno.txt", encoding="utf-8")
        documents = loader.load()

        for document in documents:
            print(f"{document}")

    except Exception as e:
        print(f"Error loading file {e}")


load_text_file()
