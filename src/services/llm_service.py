from langchain_ollama import ChatOllama


class LLMService:
    """
    Wrapper around the local Ollama model.
    """

    def __init__(self, model: str = "qwen2.5:latest"):

        self.llm = ChatOllama(
            model=model,
            temperature=0,
        )

    def generate(self, prompt: str) -> str:

        response = self.llm.invoke(prompt)

        return response.content