from src.agents.planner import PlannerAgent


def test_routes_wh_questions_to_retrieval():
    planner = PlannerAgent()

    assert planner.run("What is Retrieval-Augmented Generation?")["route"] == "retrieve"
    assert planner.run("Explain hybrid search.")["route"] == "retrieve"
    assert planner.run("Summarize this document.")["route"] == "retrieve"


def test_routes_plain_statements_direct():
    planner = PlannerAgent()

    assert planner.run("Hello!")["route"] == "direct"
    assert planner.run("Thanks for the help.")["route"] == "direct"
