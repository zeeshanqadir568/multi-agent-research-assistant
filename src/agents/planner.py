from langchain_core.messages import HumanMessage


class PlannerAgent:
    """
    Decides whether retrieval is needed.
    """

    def run(self, question: str) -> dict:

        question = question.lower()

        retrieval_keywords = [
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "explain",
            "describe",
            "compare",
            "research",
            "summarize",
        ]

        need_retrieval = any(
            keyword in question
            for keyword in retrieval_keywords
        )

        return {
            "question": question,
            "need_retrieval": need_retrieval,
        }