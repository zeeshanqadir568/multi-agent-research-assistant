"""
Document chunking utilities.

This module converts loaded documents into overlapping text chunks.
These chunks will later be embedded and indexed for retrieval.
"""

from dataclasses import dataclass

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.loader import Document


@dataclass
class Chunk:
    """
    Represents one chunk of a document.
    """

    text: str
    source: str
    chunk_id: int


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    """

    if not text.strip():
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """
    Convert loaded documents into Chunk objects.
    """

    all_chunks = []

    for document in documents:

        pieces = chunk_text(document.text)

        for index, piece in enumerate(pieces):

            all_chunks.append(
                Chunk(
                    text=piece,
                    source=document.source,
                    chunk_id=index,
                )
            )

    return all_chunks