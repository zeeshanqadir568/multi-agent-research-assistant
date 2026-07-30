"""
Embedding model wrapper.

This module loads a sentence-transformer model and converts text
into normalized embedding vectors.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Wrapper around the SentenceTransformer model.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into embedding vectors.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(np.float32)