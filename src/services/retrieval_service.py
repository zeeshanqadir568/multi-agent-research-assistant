from src.retrieval.hybrid_retriever import HybridRetriever


class RetrievalService:
    """
    Keeps one initialized HybridRetriever alive and
    exposes a simple search interface.
    """

    def __init__(self):
        self.retriever = HybridRetriever()
        self.ready = False

    def initialize(self, chunks):
        """
        Build the retrieval index once.
        """

        if self.ready:
            return

        self.retriever.build(chunks)
        self.ready = True

    def search(self, query, top_k=5):
        """
        Search the existing index.
        """

        if not self.ready:
            raise RuntimeError(
                "Retriever has not been initialized."
            )

        return self.retriever.search(query, top_k)