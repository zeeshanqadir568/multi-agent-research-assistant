from src.ingestion.loader import Document
from src.ingestion.chunker import chunk_documents
from src.retrieval.hybrid_retriever import HybridRetriever


def test_chunk_documents():

    documents = [
        Document(
            source="sample.txt",
            text="Machine learning " * 100
        )
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) > 0

    assert chunks[0].source == "sample.txt"


def test_hybrid_retriever():

    documents = [
        Document(
            source="sample.txt",
            text="""
            Machine learning is a subset of Artificial Intelligence.

            Hybrid retrieval combines dense search with BM25.

            FAISS performs vector similarity search.
            """
        )
    ]

    chunks = chunk_documents(documents)

    retriever = HybridRetriever()

    retriever.build(chunks)

    results = retriever.search(
        "What is machine learning?",
        top_k=1,
    )

    assert len(results) == 1

    assert "machine learning" in results[0].chunk.text.lower()