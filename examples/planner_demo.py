from src.agents.planner import PlannerAgent

planner = PlannerAgent()

questions = [
    "What is Retrieval-Augmented Generation?",
    "Hello!",
    "Summarize this document",
]

for q in questions:
    result = planner.run(q)

    print("=" * 60)
    print("Question:", q)
    print(result)