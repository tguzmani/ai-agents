from util.models import build_model, GPT_4_1_MINI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_community.document_loaders import DirectoryLoader, TextLoader

model = build_model(GPT_4_1_MINI)

parser = StrOutputParser()


def lazy_loader():
    try:
        loader = DirectoryLoader(
            "07_document_loaders/docs", glob="*.txt", loader_cls=TextLoader
        )

        lazy_documents = loader.lazy_load()

        for document in lazy_documents:
            print(f"Content preview: {document.page_content[:50]}...")
            print(f"Metadata: {document.metadata["source"]}")

    except Exception as e:
        print(f"Error loading doc using lazy loading: {e}")


lazy_loader()
