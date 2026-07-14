from util.models import build_embeddings, TEXT_EMBEDDING_3_SMALL
import numpy as np
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

embeddings = build_embeddings(TEXT_EMBEDDING_3_SMALL)


def embedding_caching():
    store = LocalFileStore("./10_rag/.embedding_cache")

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=store,
        namespace="10-rag-tutorial",
    )

    text = "What is reinforcement learning?"

    print("First call (from API)")
    vectors_1 = cached_embeddings.embed_documents([text])
    print(f"\tEmbedded {len(vectors_1)} documents")

    print("Second call (cached from store)")
    vectors_2 = cached_embeddings.embed_documents([text])
    print(f"\tEmbedded {len(vectors_2)} documents")

    print(f"Same vectors? {np.allclose(vectors_1[0], vectors_2[0])}")


embedding_caching()
