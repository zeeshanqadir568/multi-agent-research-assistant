from src.retrieval.hybrid_retriever import HybridRetriever


class ResearchAgent:
    """
    Retrieves relevant evidence from the knowledge base.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, question: str, top_k: int = 5):

        results = self.retriever.search(
            question,
            top_k=top_k,
        )

        return results