from util.models import build_embeddings, TEXT_EMBEDDING_3_SMALL
import numpy as np

embeddings = build_embeddings(TEXT_EMBEDDING_3_SMALL)


def basic_embeddings():
    # Simple text
    text = "What is machine learning?"

    single_embedding = embeddings.embed_query(text)

    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():
    text = [
        "What is machine learning?",
        "Explain the concept of overfitting in machine learning",
        "How does a neural network work?",
    ]

    batch_embeddings = embeddings.embed_documents(text)

    for i, embedding in enumerate(batch_embeddings):
        print(f"Text {i}, Vector dimensions: {len(embedding)}")
        print(f"\tVector dimensions: {len(embedding)}")
        print(f"\tFirst 5 values: {embedding[:5]}")
        print(f"\tNorm: {np.linalg.norm(embedding):.4f}")


# basic_embeddings()
# batch_embeddings()
