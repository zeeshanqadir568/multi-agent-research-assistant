from langgraph.graph import StateGraph, END

from src.graph.state import GraphState

from src.knowledge.knowledge_base import KnowledgeBase

from src.agents.planner import PlannerAgent
from src.agents.writer import WriterAgent
from src.agents.verifier import VerifierAgent
from src.agents.research import ResearchAgent

from src.services.llm_service import LLMService

llm_service = LLMService()

planner = PlannerAgent()
knowledge_base = KnowledgeBase()
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

    state["answer"] = (
        "Pretend an LLM answered directly.\n"
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

graph = builder.compile()