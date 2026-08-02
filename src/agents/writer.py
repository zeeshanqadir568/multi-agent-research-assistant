from src.services.llm_service import LLMService


class WriterAgent:
    """
    Generates the final answer using retrieved context.
    """

    def __init__(self, llm: LLMService):
        self.llm = llm

    def run(self, question: str, context: str) -> str:

        prompt = f"""
You are a research assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
say that the information is unavailable.

Context:
{context}

Question:
{question}

Answer:
"""

        return self.llm.generate(prompt)