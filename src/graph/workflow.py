from langgraph.graph import StateGraph, END

from src.graph.state import GraphState


def simple_node(state: GraphState):

    print("=" * 60)
    print("Simple Node Executed")
    print("=" * 60)

    state["answer"] = (
        "LangGraph is connected successfully.\n"
        f"Question received: {state['question']}"
    )

    return state


builder = StateGraph(GraphState)

builder.add_node("simple_node", simple_node)

builder.set_entry_point("simple_node")

builder.add_edge("simple_node", END)

graph = builder.compile()