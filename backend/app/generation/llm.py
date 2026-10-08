"""OpenAI-compatible LangChain chat model factory."""

from langchain_openai import ChatOpenAI

from backend.app.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create a chat model for any provider with an OpenAI-compatible API."""
    if not settings.llm_api_key.strip():
        raise RuntimeError(
            "LLM_API_KEY is not configured. Add your provider's API key to the backend .env file."
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        max_tokens=settings.llm_max_output_tokens,
        timeout=60,
        max_retries=2,
    )
