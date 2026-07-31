from src.graph.workflow import graph


result = graph.invoke(
    {
        "question": "What is Retrieval-Augmented Generation?",
        "answer": "",
    }
)

print("\n")

print(result["answer"])