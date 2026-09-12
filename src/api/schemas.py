from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, description="The question to research or answer.")


class ResearchResponse(BaseModel):
    question: str
    route: str
    plan: str
    answer: str
    sources: list[str]
