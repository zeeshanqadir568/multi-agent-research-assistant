from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL


class LLMService:
    """
    Wrapper around the local Ollama model.
    """

    def __init__(self, model: str = OLLAMA_MODEL):

        self.llm = ChatOllama(
            model=model,
            temperature=0,
        )

    def generate(self, prompt: str) -> str:

        response = self.llm.invoke(prompt)

        return response.content
