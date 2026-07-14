from util.models import build_model, GPT_4_1_MINI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)

model = build_model(GPT_4_1_MINI)

parser = StrOutputParser()


def retrieve_context(question: str) -> str:
    """Retriever falso (paso RETRIEVE del RAG).
    x
        En un RAG real recibiría la pregunta y buscaría los documentos
        relevantes en una base vectorial. Aquí ignora la pregunta y
        devuelve siempre un documento fijo sobre delfines.
    """
    return (
        "Dolphins are highly intelligent marine mammals known for their "
        "playful behavior, complex social structures, and use of "
        "echolocation to navigate and hunt."
    )


def mini_rag_chain():
    """Mini RAG: retrieve -> augment (prompt) -> generate (model)."""

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY this context.\n\n"
        "Context: {context}\n"
        "Question: {question}"
    )

    chain = (
        {
            "context": RunnableLambda(retrieve_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | parser
    )

    result = chain.invoke("What are dolphins known for?")
    print(result)


if __name__ == "__main__":
    mini_rag_chain()
