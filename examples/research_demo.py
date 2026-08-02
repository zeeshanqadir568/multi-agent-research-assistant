"""
Research Agent Demo
"""

from src.config import RAW_DATA_DIR
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agents.researcher import ResearchAgent


def main():

    print("Loading Knowledge Base...")

    documents = load_documents(RAW_DATA_DIR)

    chunks = chunk_documents(documents)

    retriever = HybridRetriever()
    retriever.build(chunks)

    researcher = ResearchAgent(retriever)

    query = "What is Retrieval-Augmented Generation?"

    print(f"\nQuery: {query}\n")

    results = researcher.run(query, top_k=3)

    for i, result in enumerate(results, start=1):

        print("=" * 50)
        print(f"Result {i}")
        print(f"Source : {result.chunk.source}")
        print(f"Score  : {result.score:.4f}")
        print("-" * 50)
        print(result.chunk.text[:200])
        print()


if __name__ == "__main__":
    main()