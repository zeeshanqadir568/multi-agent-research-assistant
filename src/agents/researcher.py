from src.retrieval.hybrid_retriever import HybridRetriever


class ResearchAgent:
    """
    Retrieves relevant evidence from the knowledge base.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(
    self,
    question: str,
    plan: str,
    top_k: int = 5,
):
        research_query = f"{question}\nResearch plan: {plan}"

        results = self.retriever.search(
            research_query,
            top_k=top_k,
        )
        return {
           "results": results,
           "context": "\n\n".join(
               result.chunk.text
               for result in results
           ),
           "sources": [
               result.chunk.source
               for result in results
    ],
}
    