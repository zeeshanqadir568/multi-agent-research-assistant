from src.agents.researcher import ResearchAgent
from src.agents.writer import WriterAgent
from src.agents.verifier import VerifierAgent

from tests.fakes import FakeKnowledgeBase, FakeLLM


def test_research_agent_returns_context_and_sources():
    agent = ResearchAgent(FakeKnowledgeBase())

    result = agent.run(question="What is hybrid retrieval?", plan="Use hybrid retrieval.", top_k=3)

    assert "hybrid" in result["context"].lower()
    assert result["sources"] == ["retrieval_systems.txt"]


def test_writer_agent_grounds_prompt_in_context():
    llm = FakeLLM(response="Hybrid retrieval combines FAISS and BM25.")
    writer = WriterAgent(llm)

    answer = writer.run(question="What is hybrid retrieval?", context="Hybrid retrieval combines dense and sparse search.")

    assert answer == "Hybrid retrieval combines FAISS and BM25."
    assert "ONLY using the provided context" in llm.prompts[0]


def test_verifier_agent_passes_answer_and_evidence_to_llm():
    llm = FakeLLM(response="Verified: hybrid retrieval combines FAISS and BM25.")
    verifier = VerifierAgent(llm)

    verified = verifier.run(
        question="What is hybrid retrieval?",
        context="Hybrid retrieval combines dense and sparse search.",
        answer="Hybrid retrieval combines FAISS and BM25.",
    )

    assert verified == "Verified: hybrid retrieval combines FAISS and BM25."
    assert "expert fact-checker" in llm.prompts[0].lower()
