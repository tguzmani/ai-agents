from pydantic import BaseModel, Field

from util.models import build_model, GEMINI_FLASH_2_5


class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")


model = build_model(GEMINI_FLASH_2_5)

structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")

print(result)
