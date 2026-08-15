"""Factory helpers for selecting an LLM backend."""

from __future__ import annotations

from typing import Any

from palace.llm.provider import BaseLLMProvider, GeminiProvider, OllamaProvider


def get_llm_provider(
    provider_type: str,
    model_name: str,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Return the configured provider implementation."""
    normalized = (provider_type or "ollama").strip().lower()
    if normalized == "ollama":
        return OllamaProvider(model_name=model_name, **kwargs)
    if normalized in {"gemini", "google", "google-gemini"}:
        return GeminiProvider(model_name=model_name, **kwargs)
    raise ValueError(f"Unsupported LLM provider: {provider_type!r}")
