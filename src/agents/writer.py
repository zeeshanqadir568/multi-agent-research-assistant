from src.services.llm_service import LLMService


class WriterAgent:
    """
    Generates the final answer using retrieved context.
    """

    def __init__(self, llm: LLMService):
        self.llm = llm

    def run(self, question: str, context: str) -> str:

        prompt = f"""
You are an AI research assistant.

Your task is to answer the user's question ONLY using the provided context.

Rules:
- Do not use outside knowledge.
- If the context does not contain enough information, clearly state that.
- Be concise but complete.
- Organize the answer into paragraphs.
- When appropriate, summarize key points using bullet points.
- Never invent facts.



Context:
{context}

Question:
{question}

Answer:
"""

        return self.llm.generate(prompt)