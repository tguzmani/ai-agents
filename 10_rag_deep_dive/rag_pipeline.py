import warnings

from util.models import (
    GEMINI_FLASH_2_5,
    TEXT_EMBEDDING_3_SMALL,
    build_embeddings,
    build_model,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging
from corpus import TECH_DOCS

load_dotenv()

embeddings = build_embeddings(TEXT_EMBEDDING_3_SMALL)
model = build_model(GEMINI_FLASH_2_5)

logging.basicConfig()
logging.getLogger("langchain_classic.retrievers").setLevel(logging.INFO)


def print_header(title: str, subtitle=None):
    print("=" * 60)
    print(title.upper())
    if subtitle:
        print(subtitle)
    print("=" * 60)


def create_base_vector_store() -> Chroma:
    """Create a basic vector store for demos."""

    return Chroma.from_documents(documents=TECH_DOCS, embedding=embeddings)


def multi_query_retriever():
    """MultiQueryRetriever generates multiple query perspectives"""

    print_header(
        title="Multi-Query Retriever",
        subtitle="Generates multiple perspective on user query",
    )

    vector_store = create_base_vector_store()

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}), llm=model
    )

    query = "What tools can I use to build AI apps?"

    print(f"\nOriginal query: {query}")

    docs = retriever.invoke(query)

    for i, doc in enumerate[Document](docs):
        print(f"\n[{i}]: {doc.page_content}")


def contextual_compression():
    print_header(title="Contextual Compression")

    vector_store = create_base_vector_store()

    compressor = LLMChainExtractor.from_llm(model)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    )

    query = "What frameworks exist for building LLM apps?"

    print(f"\nOriginal query: {query}")

    # No compression
    base_docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print(f"\n--- WITHOUT compression (full chunks) ---")

    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:150]}...\n")

    # Compression enabled
    print(f"\n--- WITH compression (full chunks) ---")

    compressed_docs = compression_retriever.invoke(query)

    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:150]}...\n")


def self_query_retriever():
    """SelfQueryRetriever splits a natural-language query into a semantic
    search + a structured metadata filter, using an LLM."""

    print_header(
        title="Self-Query Retriever",
        subtitle="Extracts a metadata filter from the natural-language query",
    )

    vector_store = create_base_vector_store()

    # Describe the metadata the LLM is allowed to filter on. Only fields that
    # exist consistently across TECH_DOCS are worth exposing here.
    metadata_field_info = [
        AttributeInfo(
            name="topic",
            description="The subject area of the document",
            type="string",
        ),
        AttributeInfo(
            name="difficulty",
            description="How advanced the material is",
            type="string",
        ),
    ]

    retriever = SelfQueryRetriever.from_llm(
        llm=model,
        vectorstore=vector_store,
        document_contents="Short descriptions of programming languages, AI tools, databases and devops tech",
        metadata_field_info=metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        # enable_limit lets the LLM also pull "top 2 ..." style limits from the query
        enable_limit=True,
        verbose=True,
    )

    # "advanced" -> filter difficulty == "advanced"; "AI" -> semantic search
    query = "What are the advanced AI topics?"

    print(f"\nOriginal query: {query}")

    docs = retriever.invoke(query)

    for i, doc in enumerate(docs):
        print(
            f"\n[{i}] (difficulty={doc.metadata.get('difficulty')}, "
            f"topic={doc.metadata.get('topic')})"
        )
        print(doc.page_content)


def hybrid_search():
    print_header(
        title="Hybrid Search", subtitle="Combines semantic search with keyword search"
    )

    vector_store = create_base_vector_store()

    bm25_retriever = BM25Retriever.from_documents(TECH_DOCS, k=3)

    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever], weights=[0.4, 0.6]
    )

    queries = [
        # BM25: keywords exactos del doc de JavaScript (React, Node, async/await)
        "async await Promises Node.js React",
        # Semántica: parafrasea LangGraph; BM25 suele priorizar LangChain por "LLM"
        "orchestrating multiple LLM actors with persistent state",
        # Split: BM25 → PostgreSQL/pgvector; semántica → vector DBs/RAG; ensemble → ambos
        "relational database with vector similarity for AI search",
    ]

    for query in queries:
        print(f"\nOriginal query: {query}")

        bm25_docs = bm25_retriever.invoke(query)
        semantic_docs = semantic_retriever.invoke(query)
        ensemble_docs = ensemble_retriever.invoke(query)

        for i, doc in enumerate(bm25_docs):
            print(f"\n[{i}] (BM25)")
            print(doc.page_content)

        for i, doc in enumerate(semantic_docs):
            print(f"\n[{i}] (Semantic)")
            print(doc.page_content)

        for i, doc in enumerate(ensemble_docs):
            print(f"\n[{i}] (Ensemble)")
            print(doc.page_content)


def parent_document_retriever():
    print_header(
        title="Parent Document Retriever",
        subtitle="Retrieves the parent document of a given document",
    )

    vector_store = Chroma(
        collection_name="parent_document_retriever",
        embedding_function=embeddings,
    )

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    store = InMemoryStore()

    parent_retriever = ParentDocumentRetriever(
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        vectorstore=vector_store,
        docstore=store,
        search_kwargs={"k": 2},
    )

    parent_retriever.add_documents(TECH_DOCS)

    query = "What is the parent document of the document about PostgreSQL?"

    child_docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(query)
    for i, doc in enumerate(child_docs):
        print(f"\n[{i}] (Child)")
        print(doc.page_content)

    parent_docs = parent_retriever.invoke(query)
    for i, doc in enumerate(parent_docs):
        print(f"\n[{i}] (Parent)")
        print(doc.page_content)


if __name__ == "__main__":
    # multi_query_retriever()
    # contextual_compression()
    # self_query_retriever()
    # hybrid_search()
    parent_document_retriever()
