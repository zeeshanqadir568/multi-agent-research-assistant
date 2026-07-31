from typing import Literal
from typing import TypedDict


class GraphState(TypedDict):
    question: str
    route: Literal["retrieve", "direct"]
    plan: str
    context: str
    answer: str