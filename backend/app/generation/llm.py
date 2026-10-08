"""OpenAI-compatible LangChain chat model factory."""

from langchain_openai import ChatOpenAI

from backend.app.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the backend's configured OpenAI-compatible chat model."""
    if not settings.openai_api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to the .env file to generate answers."
        )
    return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            timeout=60,
            max_retries=2,
    )
