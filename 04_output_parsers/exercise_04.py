from enum import Enum

from pydantic import BaseModel, Field

from util.models import build_model, GEMINI_FLASH_2_5


class Role(str, Enum):
    director = "director"
    actor = "actor"


class Person(BaseModel):
    name: str = Field(description="Persons's full name")
    role: Role = Field(description="Person role in the movie: director or actor")


class Movie(BaseModel):
    title: str = Field(description="The movie title")
    year: str = Field(description="The year the movie was released")
    director: Person = Field(description="Movie director")
    actors: list[Person] = Field(description="The actors involved in the movie")
    genre: str = Field(description="Movie genre")
    rating: int = Field(description="Movie rating")


def exercise_structure_extraction():
    """
    Exerxise: Create a schema and chain that extracts

    - Movie title
    - Year released
    - Director
    - Main actors (list)
    - Genre
    - Rating (1-10)
    """

    model = build_model(GEMINI_FLASH_2_5).with_structured_output(Movie)

    result = model.invoke(
        "Get release years, director, main actors, genre and average "
        "rating for the information for the movie: The Green Mile"
    )

    for key, value in result:
        print(f"{key}: {value}\n")


if __name__ == "__main__":
    exercise_structure_extraction()
