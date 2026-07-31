from langgraph.graph import StateGraph, END

from src.graph.state import GraphState

from src.knowledge.knowledge_base import KnowledgeBase

knowledge_base = KnowledgeBase()


def planner_node(state: GraphState):

    print("\n=== Planner ===")

    question = state["question"].lower()

    if "document" in question or "pdf" in question:

        state["route"] = "retrieve"

        state["plan"] = "Use hybrid retrieval."

    else:

        state["route"] = "direct"

        state["plan"] = "Answer directly."

    return state


def retrieval_node(state: GraphState):

    print("\n=== Retrieval ===")

    results = knowledge_base.search(
        state["question"],
        top_k=3,
    )

    context = "\n\n".join(
        result.chunk.text
        for result in results
    )

    state["context"] = context
    state["answer"] = context
    

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

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "planner",
    route_question,
    {
        "retrieve": "retrieval",
        "direct": "direct",
    },
)

builder.add_edge("retrieval", END)

builder.add_edge("direct", END)

graph = builder.compile()