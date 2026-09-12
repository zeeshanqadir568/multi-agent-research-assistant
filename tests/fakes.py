"""
Fakes used across the test suite so the graph, agents, and API can be
exercised without a live Ollama server or a real embedding index.
"""

from src.ingestion.chunker import Chunk
from src.retrieval.hybrid_retriever import RetrievalResult


class FakeLLM:
    """A scripted stand-in for LLMService. No network calls."""

    def __init__(self, response: str = "This is a fake, evidence-grounded answer."):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeKnowledgeBase:
    """Returns canned retrieval results instead of running FAISS/BM25."""

    def __init__(self, results: list[RetrievalResult] | None = None):
        if results is None:
            chunk = Chunk(
                text="Hybrid retrieval combines dense (FAISS) and sparse (BM25) search.",
                source="retrieval_systems.txt",
                chunk_id=0,
            )
            results = [
                RetrievalResult(chunk=chunk, score=1.0, dense_score=1.0, bm25_score=1.0)
            ]
        self.results = results

    def search(self, query: str, top_k: int = 5):
        return self.results[:top_k]
