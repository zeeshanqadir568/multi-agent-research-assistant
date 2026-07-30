"""
Simple demo for the hybrid retrieval pipeline.
"""

from src.config import RAW_DATA_DIR
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.hybrid_retriever import HybridRetriever


def main():

    print("=" * 60)
    print("Loading documents...")
    print("=" * 60)

    documents = load_documents(RAW_DATA_DIR)

    print(f"Loaded {len(documents)} document(s).\n")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunk(s).\n")

    retriever = HybridRetriever()

    print("Building retrieval index...\n")

    retriever.build(chunks)

    queries = [
        "What is machine learning?",
        "What is hybrid retrieval?",
        "How does BM25 work?"
    ]

    for query in queries:

        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        results = retriever.search(query, top_k=3)

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print(f"Source      : {result.chunk.source}")
            print(f"Chunk ID    : {result.chunk.chunk_id}")
            print(f"Score       : {result.score:.4f}")
            print(f"Dense Score : {result.dense_score:.4f}")
            print(f"BM25 Score  : {result.bm25_score:.4f}")
            print("-" * 50)
            print(result.chunk.text[:200])
            print()

    print("=" * 60)
    print("Demo completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()