"""OpenAI-compatible LangChain chat model factory."""

from langchain_openai import ChatOpenAI

from app.config import Settings


def create_chat_model(settings: Settings, model: str | None = None) -> ChatOpenAI:
    """Create a chat model without exposing credentials in logs."""
    if not settings.openai_api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to the .env file to generate answers."
        )
    options: dict[str, str | float] = {
        "model": model or settings.llm_model,
        "api_key": settings.openai_api_key,
        "temperature": 0,
        "timeout": 60,
        "max_retries": 2,
    }
    if settings.openai_base_url:
        options["base_url"] = settings.openai_base_url
    return ChatOpenAI(**options)
