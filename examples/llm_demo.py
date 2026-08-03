from src.services.llm_service import LLMService


def main():

    llm = LLMService()

    response = llm.generate(
        "Explain Retrieval-Augmented Generation in two sentences."
    )

    print("\nResponse:\n")
    print(response)


if __name__ == "__main__":
    main() 

    