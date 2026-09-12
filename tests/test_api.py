from fastapi.testclient import TestClient

from src.api.main import create_app
from src.graph.workflow import build_graph

from tests.fakes import FakeKnowledgeBase, FakeLLM


def make_client(response: str = "Hybrid retrieval combines FAISS and BM25.") -> TestClient:
    graph = build_graph(llm_service=FakeLLM(response=response), knowledge_base=FakeKnowledgeBase())
    app = create_app(graph=graph)
    return TestClient(app)


def test_health():
    with make_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_retrieval_path():
    with make_client(response="Hybrid retrieval combines FAISS and BM25.") as client:
        response = client.post("/research", json={"question": "What is hybrid retrieval?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "retrieve"
    assert body["sources"] == ["retrieval_systems.txt"]
    assert body["answer"] == "Hybrid retrieval combines FAISS and BM25."


def test_research_direct_path():
    with make_client(response="Hi there!") as client:
        response = client.post("/research", json={"question": "Hello!"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "direct"
    assert body["sources"] == []


def test_research_rejects_empty_question():
    with make_client() as client:
        response = client.post("/research", json={"question": ""})

    assert response.status_code == 422
