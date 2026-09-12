from langgraph.graph import StateGraph, END

from src.graph.state import GraphState

from src.knowledge.knowledge_base import KnowledgeBase

from src.agents.planner import PlannerAgent
from src.agents.writer import WriterAgent
from src.agents.verifier import VerifierAgent
from src.agents.researcher import ResearchAgent

from src.services.llm_service import LLMService


def build_graph(llm_service=None, knowledge_base=None):
    """
    Assemble the Planner -> Research -> Writer -> Verifier graph.

    Accepts an injected llm_service / knowledge_base so callers (the API,
    tests) can run the graph without a live Ollama server or a real
    embedding index. Defaults to the real services when not provided.
    """

    llm_service = llm_service or LLMService()
    knowledge_base = knowledge_base or KnowledgeBase()

    planner = PlannerAgent()
    research = ResearchAgent(knowledge_base)
    verifier = VerifierAgent(llm_service)
    writer = WriterAgent(llm_service)

    def planner_node(state: GraphState):

        print("\n=== Planner ===")

        decision = planner.run(state["question"])

        state["route"] = decision["route"]
        state["plan"] = (
            "Use hybrid retrieval."
            if state["route"] == "retrieve"
            else "Answer directly."
        )

        return state

    def retrieval_node(state: GraphState):

        print("\n=== Research ===")

        research_result = research.run(
            question=state["question"],
            plan=state["plan"],
            top_k=3,
        )

        state["context"] = research_result["context"]
        state["sources"] = research_result["sources"]

        return state

    def writer_node(state: GraphState):

        print("\n=== Writer ===")

        state["answer"] = writer.run(
            question=state["question"],
            context=state["context"],
        )

        return state

    def verifier_node(state: GraphState):

        print("\n=== Verifier ===")

        verified_answer = verifier.run(
            question=state["question"],
            context=state["context"],
            answer=state["answer"],
        )

        state["answer"] = verified_answer

        return state

    def direct_node(state: GraphState):

        print("\n=== Direct ===")

        state["sources"] = []
        state["answer"] = llm_service.generate(
            "Answer the following question directly and concisely, "
            "using your own knowledge. If you are not confident in the "
            "answer, say so explicitly.\n\n"
            f"Question: {state['question']}"
        )

        return state

    def route_question(state: GraphState):

        return state["route"]

    builder = StateGraph(GraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("direct", direct_node)
    builder.add_node("writer", writer_node)
    builder.add_node("verifier", verifier_node)

    builder.set_entry_point("planner")

    builder.add_conditional_edges(
        "planner",
        route_question,
        {
            "retrieve": "retrieval",
            "direct": "direct",
        },
    )

    builder.add_edge("retrieval", "writer")
    builder.add_edge("writer", "verifier")
    builder.add_edge("verifier", END)
    builder.add_edge("direct", END)

    return builder.compile()
