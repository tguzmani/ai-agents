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


def create_knowledge_base():
    """Create a vector store from kwnoledge base"""

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )
