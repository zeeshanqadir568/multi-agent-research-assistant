from src.graph.workflow import graph


questions = [
    "What is Retrieval-Augmented Generation?",
    "Summarize this PDF document.",
]

for question in questions:

    print("\n" + "=" * 70)

    result = graph.invoke(
        {
            "question": question,
            "route": "",
            "plan": "",
            "answer": "",
        }
    )

    print("\nFinal State")

    print(result)