# multi-agent-research-assistant
# Multi-Agent Research & Report Assistant

A production-style AI system that performs document research using multiple
specialized agents. Instead of relying on a single Retrieval-Augmented
Generation (RAG) pipeline, the system plans research, retrieves evidence,
verifies claims, writes structured sections, and produces a cited final report.

## 🚀 Features

- Multi-agent workflow using LangGraph — Planner → Research → Writer → Verifier
- Hybrid retrieval: FAISS (dense) + BM25 (sparse) with score fusion
- Evidence-verification pass to reduce hallucinations
- Local LLM support via Ollama (through LangChain)
- Modular, extensible layout — `agents/`, `graph/`, `retrieval/`, `ingestion/`, `services/`

### Planned / not yet implemented

- **FastAPI backend** — `src/api/` is currently a stub; the workflow runs from `examples/` scripts
- **n8n automation hooks**

## 🛠️ Tech Stack

- Python
- LangGraph
- LangChain
- FAISS
- BM25 (`rank-bm25`)
- Sentence Transformers
- Ollama
- Pytest

## ▶️ Running it

```bash
pip install -r requirements.txt
# hybrid retriever only:
python -m examples.demo
# full LangGraph workflow (needs a local Ollama model):
python -m examples.langgraph_demo
```

## 📂 Project Status

🚧 Active development — Phase 1 (project foundation). The LangGraph workflow and
the hybrid retriever run end to end from the `examples/` scripts; the HTTP API
and automation layer are not built yet.

## 📜 License

MIT License
