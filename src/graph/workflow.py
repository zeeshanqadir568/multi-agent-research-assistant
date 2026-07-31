from langgraph.graph import StateGraph, END

from src.graph.state import GraphState


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

    state["answer"] = (
        "Pretend we searched the vector database.\n"
        f"Question: {state['question']}"
    )

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