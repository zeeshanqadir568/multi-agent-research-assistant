class PlannerAgent:
    """
    Decides how the graph should route a question.
    """

    RETRIEVAL_KEYWORDS = {
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
        "document",
        "pdf",
    }

    def run(self, question: str) -> dict:

        q = question.lower()

        need_retrieval = any(
            word in q
            for word in self.RETRIEVAL_KEYWORDS
        )

        return {
            "route": "retrieve" if need_retrieval else "direct"
        }