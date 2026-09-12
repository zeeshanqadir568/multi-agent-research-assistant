# multi-agent-research-assistant

## Multi-Agent Research & Report Assistant

A production-style AI system that performs document research using multiple
specialized agents. Instead of relying on a single Retrieval-Augmented
Generation (RAG) pipeline, the system plans research, retrieves evidence,
verifies claims, writes structured sections, and produces a cited final report.

## 🚀 Features

- Multi-agent workflow using LangGraph — Planner → Research → Writer → Verifier
- Hybrid retrieval: FAISS (dense) + BM25 (sparse) with score fusion
- Evidence-verification pass to reduce hallucinations
- Local LLM support via Ollama (through LangChain)
- FastAPI backend (`POST /research`) — the same graph the CLI examples run, served over HTTP with a Swagger UI at `/docs`
- Modular, extensible layout — `agents/`, `graph/`, `retrieval/`, `ingestion/`, `services/`, `api/`
- 13 tests, no network for the graph/agent/API suite (a scripted LLM and a fake knowledge base stand in for Ollama and the embedding index)

### Planned / not yet implemented

- **n8n automation hooks**

## 🛠️ Tech Stack

- Python
- LangGraph
- LangChain
- FastAPI
- FAISS
- BM25 (`rank-bm25`)
- Sentence Transformers
- Ollama
- Pytest

## ▶️ Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # optional — set OLLAMA_MODEL to a model you've pulled

# hybrid retriever only:
python -m examples.demo

# full LangGraph workflow from the CLI (needs a local Ollama model):
python -m examples.langgraph_demo

# HTTP API (needs a local Ollama model):
uvicorn src.api.main:app --reload
# then open http://127.0.0.1:8000/docs
```

`OLLAMA_MODEL` defaults to `qwen2.5:latest`. Set it in `.env` to whatever
model you already have pulled (`ollama list`), e.g. `OLLAMA_MODEL=qwen3:8b`.

## 📡 API

Base URL: `http://127.0.0.1:8000`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/research` | Run the full Planner → Research → Writer → Verifier graph on one question |
| GET | `/docs` | Swagger UI |

`POST /research` request:

```json
{ "question": "What is hybrid retrieval?" }
```

Response:

```json
{
  "question": "What is hybrid retrieval?",
  "route": "retrieve",
  "plan": "Use hybrid retrieval.",
  "answer": "Hybrid retrieval is a method that combines dense and sparse retrieval approaches...",
  "sources": ["retrieval_systems.txt", "ml_basics.txt"]
}
```

Questions that don't need the knowledge base (`route: "direct"`) skip retrieval
and are answered by the LLM directly, with an empty `sources` list.

## 📂 Project Status

🚧 Active development. The LangGraph workflow, the hybrid retriever, and the
HTTP API all run end to end — verified locally against a real Ollama model.
The automation layer (n8n) is not built yet.

## 🧪 Testing

```bash
pytest
```

13 tests, no network: the planner, researcher, writer, verifier, graph
routing (both the retrieval and direct paths), and every API endpoint are
covered using a scripted `FakeLLM` and a `FakeKnowledgeBase` (see
`tests/fakes.py`) — no Ollama server or embedding model required to run the
suite.

## 📜 License

MIT License
