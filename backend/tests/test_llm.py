"""Tests for configurable, OpenAI-compatible LLM settings."""

from unittest.mock import Mock, patch

import pytest

from backend.app.config import Settings
from backend.app.generation.llm import create_chat_model


def test_chat_model_uses_configured_provider_and_token_limit():
    settings = Settings(
        _env_file=None,
        llm_api_key="provider-key",
        llm_base_url="https://provider.example/v1",
        llm_model="provider-model",
        llm_max_output_tokens=400,
    )
    model = Mock()

    with patch("backend.app.generation.llm.ChatOpenAI", return_value=model) as factory:
        assert create_chat_model(settings) is model

    factory.assert_called_once_with(
        model="provider-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        temperature=0,
        max_tokens=400,
        timeout=60,
        max_retries=2,
    )


def test_legacy_openai_environment_names_remain_supported():
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="legacy-key",
        OPENAI_BASE_URL="https://legacy.example/v1",
    )
    assert settings.llm_api_key == "legacy-key"
    assert settings.llm_base_url == "https://legacy.example/v1"


def test_chat_model_requires_provider_api_key():
    settings = Settings(_env_file=None, llm_api_key="")
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        create_chat_model(settings)
