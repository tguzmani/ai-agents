import logging
import warnings

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.storage import InMemoryStore
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpus import TECH_DOCS
from util.models import (
    GEMINI_FLASH_2_5,
    TEXT_EMBEDDING_3_SMALL,
    build_embeddings,
    build_model,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


load_dotenv()

embeddings = build_embeddings(TEXT_EMBEDDING_3_SMALL)
model = build_model(GEMINI_FLASH_2_5)
parser = StrOutputParser()


def print_header(title: str, subtitle=None):
    print("=" * 60)
    print(title.upper())
    if subtitle:
        print(subtitle)
    print("=" * 60)


def create_base_vector_store() -> Chroma:
    """Create a basic vector store for demos."""

    return Chroma.from_documents(documents=TECH_DOCS, embedding=embeddings)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def advanced_rag_chain():
    print_header(
        title="Advanced RAG Chain",
    )

    vector_store = create_base_vector_store()

    # Multi-query retriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), llm=model
    )

    # Compression to focus on relevant info
    compressor = LLMChainExtractor.from_llm(
        llm=model,
    )

    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever,
    )

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context: {context}\n\nQuestion: {question}\n\nAnswer:"
        "Be specific and cite with page numbers from the context if possible."
        "If you don't know the answer, say 'I don't know'."
    )

    context_chain = advanced_retriever | format_docs

    rag_chain = (
        {"context": context_chain, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    response = rag_chain.invoke({"question": "What is the main topic of the document?"})
    print(response)


if __name__ == "__main__":
    advanced_rag_chain()
