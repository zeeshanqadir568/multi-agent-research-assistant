"""
Knowledge Base

Responsible for:

- Loading documents
- Chunking
- Building the retrieval index

This happens once when the application starts.
"""

from src.config import RAW_DATA_DIR
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.hybrid_retriever import HybridRetriever


class KnowledgeBase:

    def __init__(self):

        print("Loading Knowledge Base...")

        documents = load_documents(RAW_DATA_DIR)

        chunks = chunk_documents(documents)

        self.retriever = HybridRetriever()

        self.retriever.build(chunks)

        print(f"Knowledge Base Ready ({len(chunks)} chunks)")

    def search(self, query: str, top_k: int = 5):

        return self.retriever.search(query, top_k)