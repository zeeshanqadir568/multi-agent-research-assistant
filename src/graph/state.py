from typing import TypedDict


class GraphState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    """

    question: str
    answer: str