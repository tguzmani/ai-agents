"""
QA Smart Bot
"""

from util.models import build_model, GPT_4_1_MINI
from qa_response import QAResponse, ConfidenceLevel
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable, Client


class SmartQABot:
    def __init__(self, model_name: str = GPT_4_1_MINI, temperature: float = 0.3):
        self.model = build_model(model_name, temperature).with_structured_output(
            QAResponse
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a knwoledgeable Q&A Assistance"),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.model

    @traceable(name="ask", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            return QAResponse(
                answer="I'm sorry, I could not process your question at this time...",
                confidence=ConfidenceLevel.low,
                reasoning=str(e),
                follow_up_questions=["Could you please try again later"],
                sources_needed=False,
            )

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: list[str]) -> list[QAResponse]:
        inputs = [{"question": question} for question in questions]
        return self.chain.batch(inputs)

    def flush(self):
        Client().flush()
