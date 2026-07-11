from util.models import build_embeddings, TEXT_EMBEDDING_3_SMALL
import numpy as np

embeddings = build_embeddings(TEXT_EMBEDDING_3_SMALL)


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    vector_b_norm = np.linalg.norm(vector_b)
    vector_a_norm = np.linalg.norm(vector_a)

    return dot_product / (vector_a_norm * vector_b_norm)


def similarity_search():
    docs = [
        "Python is a programming language",
        "Javascript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "Tell me about AI and machine learning"

    doc_embeddings = embeddings.embed_documents(docs)
    query_embedding = embeddings.embed_query(query)

    similarities = [
        cosine_similarity(query_embedding, doc_embedding)
        for doc_embedding in doc_embeddings
    ]

    sorted_similarities = sorted(
        zip(docs, similarities), key=lambda x: x[1], reverse=True
    )

    # Rank documents by similarity

    print(f"Query: {query}\n")
    print("Ranked by similarity")

    for doc, score in sorted_similarities:
        print(f"\t{score:.4f}: {doc}")


similarity_search()
