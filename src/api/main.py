from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schemas import ResearchRequest, ResearchResponse
from src.graph.workflow import build_graph


def create_app(graph=None) -> FastAPI:
    """
    Build the FastAPI app. Pass a pre-built `graph` (e.g. one wired to a
    fake LLM / knowledge base) to avoid touching Ollama or the embedding
    index — this is how tests run the API with no network calls.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph = graph if graph is not None else build_graph()
        yield

    app = FastAPI(
        title="Multi-Agent Research Assistant",
        description=(
            "Planner -> Research -> Writer -> Verifier over hybrid "
            "(FAISS + BM25) retrieval, with an evidence-verification pass."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/research", response_model=ResearchResponse)
    def research(request: ResearchRequest):
        result = app.state.graph.invoke(
            {
                "question": request.question,
                "route": "",
                "plan": "",
                "context": "",
                "sources": [],
                "answer": "",
            }
        )

        return ResearchResponse(
            question=request.question,
            route=result["route"],
            plan=result["plan"],
            answer=result["answer"],
            sources=result.get("sources", []),
        )

    return app


app = create_app()
