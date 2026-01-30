"""
LLM helper for SelfEvolvedReLoop.

Supports OpenAI-compatible endpoints. Primary env vars:
- OR_LLM_BASE_URL
- OR_LLM_API_KEY
- OR_LLM_MODEL (optional)
Falls back to OPENAI_API_KEY/OPENAI_MODEL for backward compatibility.
"""
from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI


def _env_api_key() -> Optional[str]:
    return os.environ.get("OR_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _env_base_url() -> Optional[str]:
    return os.environ.get("OR_LLM_BASE_URL")


def _env_model(model_name: Optional[str]) -> str:
    return (
        model_name
        or os.environ.get("OR_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o-mini"
    )


def get_llm_client(model_name: Optional[str] = None, temperature: float = 0.0):
    """
    Return a ChatOpenAI client or None when no API key is present.
    """
    api_key = _env_api_key()
    if not api_key:
        return None
    base_url = _env_base_url()
    name = _env_model(model_name)
    return ChatOpenAI(model=name, temperature=temperature, api_key=api_key, base_url=base_url)


# backward compatible alias
def get_llm(model_name: Optional[str] = None, temperature: float = 0.0):
    return get_llm_client(model_name=model_name, temperature=temperature)


__all__ = ["get_llm", "get_llm_client"]
