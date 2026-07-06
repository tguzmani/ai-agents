from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    confidence: ConfidenceLevel = Field(
        description="Confidence level: high, medium or low"
    )
    reasoning: str = Field(description="The reasoning behind the answer provided")
    follow_up_questions: list[str] = Field(
        description="A list of follow-up questions related to the topic",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Indicates whether sources are needed for the answer", default=False
    )
