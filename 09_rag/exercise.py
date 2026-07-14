from util.models import build_embeddings, TEXT_EMBEDDING_3_SMALL
import numpy as np
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = build_embeddings(TEXT_EMBEDDING_3_SMALL)

DOCUMENTS = [
    Document(
        page_content=(
            "Embeddear un texto es una función determinista: para un mismo par "
            "(modelo, texto) siempre sale el mismo vector. Como no hay "
            "aleatoriedad, cachear los embeddings nunca cambia el resultado, "
            "solo evita recomputar y pagar la API de nuevo."
        ),
        metadata={"source": "notas/caching.md", "topic": "embedding-caching"},
    ),
    Document(
        page_content=(
            "El caching de embeddings conviene cuando el hit rate es alto, como "
            "en los documentos de un RAG que se re-indexan entre corridas. No "
            "conviene para queries de usuario, que son casi siempre únicas; por "
            "eso CacheBackedEmbeddings solo cachea embed_documents por defecto."
        ),
        metadata={"source": "notas/caching.md", "topic": "embedding-caching"},
    ),
    Document(
        page_content=(
            "Una base de datos vectorial indexa por proximidad geométrica en un "
            "espacio de muchas dimensiones, no por valores exactos. Usa índices "
            "ANN como HNSW para bajar la búsqueda de vecinos más cercanos de "
            "O(n) a aproximadamente O(log n), a cambio de ser aproximada."
        ),
        metadata={"source": "notas/vector-db.md", "topic": "vector-databases"},
    ),
    Document(
        page_content=(
            "Chroma es el SQLite de las vector DBs: embebida en el proceso, cero "
            "setup y persistencia en una carpeta local, ideal para aprender y "
            "prototipar. Qdrant es el Postgres: un servidor dedicado en Rust, "
            "con filtrado de metadata potente y escala a millones de vectores."
        ),
        metadata={"source": "notas/vector-db.md", "topic": "chroma-vs-qdrant"},
    ),
    Document(
        page_content=(
            "La similitud coseno mide el ángulo entre dos vectores: "
            "cos(theta) = (A·B) / (||A|| · ||B||). Es la métrica típica para "
            "comparar embeddings porque ignora la magnitud y se enfoca en la "
            "orientación, es decir, en el significado más que en la longitud."
        ),
        metadata={"source": "notas/similaridad.md", "topic": "similarity-search"},
    ),
    Document(
        page_content=(
            "RAG se usa cuando necesitas conocimiento factual actualizable o de "
            "dominio; fine-tuning cuando quieres cambiar comportamiento, estilo "
            "o formato. La inyección de contexto fija sirve para reglas estables "
            "y baratas: siempre empieza por lo más simple y escala si no alcanza."
        ),
        metadata={"source": "notas/genai.md", "topic": "rag-vs-finetuning"},
    ),
    Document(
        page_content=(
            "Un agent es un LLM que decide qué acciones tomar en un loop, usando "
            "herramientas y observando resultados para planear el siguiente "
            "paso. La diferencia clave con una chain es que el control de flujo "
            "lo decide el modelo, no un pipeline fijo definido de antemano."
        ),
        metadata={"source": "notas/genai.md", "topic": "agents"},
    ),
    Document(
        page_content=(
            "Groundedness y faithfulness miden si cada afirmación de la "
            "respuesta está respaldada por el contexto recuperado, sin inventar "
            "ni contradecir la fuente. Son las métricas centrales contra "
            "alucinaciones en un pipeline RAG."
        ),
        metadata={"source": "notas/evaluacion.md", "topic": "groundedness"},
    ),
    Document(
        page_content=(
            "En retrieval, precision es qué fracción de los chunks recuperados "
            "era relevante, y recall es qué fracción de los relevantes lograste "
            "recuperar. Hay tensión entre ambas: normalmente priorizas recall al "
            "recuperar amplio y luego afinas precision con un paso de reranking."
        ),
        metadata={"source": "notas/evaluacion.md", "topic": "precision-recall"},
    ),
    Document(
        page_content=(
            "RAGAS es un framework que automatiza métricas de RAG como "
            "faithfulness, answer relevancy, context precision y context recall, "
            "usando LLM-as-a-judge por debajo. Un juez LLM evalúa salidas de "
            "otro LLM con una rúbrica, y se valida contra ejemplos etiquetados."
        ),
        metadata={"source": "notas/evaluacion.md", "topic": "ragas"},
    ),
]

vector_store = Chroma(
    # documents=DOCUMENTS,
    embedding_function=embeddings,
    persist_directory="./10_rag/chroma",
)


def chroma_basics():

    count = vector_store._collection.count()

    print(f"Vectore store created with {count} collections and persisted")

    query = "Que es rag?"

    results = vector_store.similarity_search(query, k=2)

    print(f"Top results for query '{query}'")
    for i, doc in enumerate(results):
        print(f"Result {i}: {doc.page_content} [source: {doc.metadata['source']}]")


def similarity_search_with_scores():
    query = "Que es una base de datos vectorial?"

    results = vector_store.similarity_search_with_score(query, k=3)

    print(f"Top results for query '{query}'")
    for i, (doc, score) in enumerate(results):
        print(
            f"Result {i}: {doc.page_content}\n[source: {doc.metadata['source']}, score: {score:.2f}]\n"
        )


def as_retriever():
    # this is a Runnable!
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )

    query = "How do I build AI apps?"

    docs = retriever.invoke(query)

    print(f"Top results for query '{query}'")
    for i, doc in enumerate(docs):
        print(f"Result {i}: {doc.page_content} [source: {doc.metadata['source']}]")

    # MMR: max marginal relevance, gives diverse results

    mmr_query = "recall?"

    mmr_retrievr = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
    )

    mmr_docs = mmr_retrievr.invoke(mmr_query)

    print(f"Top results for query '{mmr_query}'")
    for i, doc in enumerate(mmr_docs):
        print(f"Result {i}: {doc.page_content} [source: {doc.metadata['source']}]")


# similarity_search_with_scores()
as_retriever()
