import os
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq

load_dotenv(find_dotenv(usecwd=True))


def validate_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Create a .env file in the project root or set the environment variable before running the app.\n"
            "Example: GROQ_API_KEY=your_api_key_here"
        )
    return api_key


def get_llm(model: str, temperature: float = 0, **kwargs) -> ChatGroq:
    api_key = kwargs.pop("api_key", None) or validate_api_key()
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        **kwargs,
    )
