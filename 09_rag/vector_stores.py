import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from util.models import build_embeddings, TEXT_EMBEDDING_3_SMALL
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
import numpy as np
from sys import exit

from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = build_embeddings(TEXT_EMBEDDING_3_SMALL)


def get_documents():
    loader = TextLoader("08_chunking/docs/sistema_solar.txt", encoding="utf-8")
    documents = loader.load()

    return documents


def get_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents=documents)

    return chunks


def create_chroma_store(documents):
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./09_rag/exercise_chroma",
    )

    return vector_store


def get_chroma_store():
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./09_rag/exercise_chroma",
    )

    return vector_store


def retrieve(vector_store: Chroma, query: str):
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
    )

    retrived_docs = retriever.invoke(query)

    print(f"Top results for query '{query}'")
    for i, doc in enumerate(retrived_docs):
        print(f"Result {i}: {doc.page_content} [source: {doc.metadata['source']}]")


if __name__ == "__main__":
    vector_store = get_chroma_store()

    is_store_empty = get_chroma_store()._collection.count() == 0

    query = "Cual es el planeta mas grande del sistema solar?"

    if is_store_empty:
        print("Vector store is empty, will create a new one")
        documents = get_documents()
        chunks = get_chunks(documents=documents)
        vector_store = create_chroma_store(documents=chunks)
    else:
        print("Vector store has documents, will load it")
        vector_store = get_chroma_store()

    retrieve(vector_store=vector_store, query=query)
