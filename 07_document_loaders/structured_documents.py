from util.models import build_model, GPT_4_1_MINI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

model = build_model(GPT_4_1_MINI)

parser = StrOutputParser()


def doc_structure():
    doc = Document(
        page_content="This is a sample document",
        metadata={"source": "sample.txt", "author": "Tom Guz"},
    )
