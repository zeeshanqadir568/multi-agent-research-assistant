from src.services.llm_service import LLMService


class VerifierAgent:
    """
    Reviews the generated answer against retrieved evidence.
    """

    def __init__(self, llm: LLMService):
        self.llm = llm

    def run(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> str:

        prompt = f"""
You are an expert fact-checker.

Question:
{question}

Retrieved Evidence:
{context}

Generated Answer:
{answer}

Evaluate the answer using ONLY the retrieved evidence.

If every important claim is supported,
respond with the answer unchanged.

If something is unsupported,
rewrite the answer so that every statement
is supported by the retrieved evidence.

Do not invent facts.
Do not use outside knowledge.
Return only the final verified answer.
"""

        return self.llm.generate(prompt)