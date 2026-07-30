"""
Hybrid retrieval using FAISS (dense search) + BM25 (sparse search).
"""

from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import HYBRID_ALPHA
from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import EmbeddingModel


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_score: float
    bm25_score: float


class HybridRetriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.index = None
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_model.encode(texts)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:

        query_embedding = self.embedding_model.encode([query])

        dense_scores, dense_indices = self.index.search(query_embedding, len(self.chunks))

        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        tokenized_query = query.lower().split()

        bm25_scores = np.array(
            self.bm25.get_scores(tokenized_query),
            dtype=np.float32,
        )

        dense_norm = self._normalize(dense_scores)

        bm25_norm = self._normalize(bm25_scores)

        results = []

        for rank, chunk_index in enumerate(dense_indices):

            combined = (
                HYBRID_ALPHA * dense_norm[rank]
                + (1 - HYBRID_ALPHA) * bm25_norm[chunk_index]
            )

            results.append(
                RetrievalResult(
                    chunk=self.chunks[chunk_index],
                    score=float(combined),
                    dense_score=float(dense_scores[rank]),
                    bm25_score=float(bm25_scores[chunk_index]),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:

        minimum = scores.min()
        maximum = scores.max()

        if maximum - minimum == 0:
            return np.ones_like(scores)

        return (scores - minimum) / (maximum - minimum)