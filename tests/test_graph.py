from src.graph.workflow import build_graph

from tests.fakes import FakeKnowledgeBase, FakeLLM


def make_initial_state(question: str) -> dict:
    return {
        "question": question,
        "route": "",
        "plan": "",
        "context": "",
        "sources": [],
        "answer": "",
    }


def test_retrieval_path_uses_context_and_returns_sources():
    llm = FakeLLM(response="Hybrid retrieval combines FAISS and BM25.")
    graph = build_graph(llm_service=llm, knowledge_base=FakeKnowledgeBase())

    result = graph.invoke(make_initial_state("What is hybrid retrieval?"))

    assert result["route"] == "retrieve"
    assert result["sources"] == ["retrieval_systems.txt"]
    assert result["answer"] == "Hybrid retrieval combines FAISS and BM25."
    # Writer then Verifier both ran against the LLM.
    assert len(llm.prompts) == 2


def test_direct_path_skips_retrieval_and_has_no_sources():
    llm = FakeLLM(response="Hi there!")
    graph = build_graph(llm_service=llm, knowledge_base=FakeKnowledgeBase())

    result = graph.invoke(make_initial_state("Hello!"))

    assert result["route"] == "direct"
    assert result["sources"] == []
    assert result["answer"] == "Hi there!"
    assert len(llm.prompts) == 1
