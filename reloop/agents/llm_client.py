from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


@dataclass
class LLMResponse:
    content: str
    raw: Any
    usage: Dict[str, Any]


class LLMClient:
    """Abstract adapter used by the agent stack."""

    def complete(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:  # pragma: no cover - interface only
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic mock that simply echos the last human message."""

    def complete(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        last_human = ""
        for msg in reversed(messages):
            if getattr(msg, "type", getattr(msg, "role", "")) == "human":
                last_human = msg.content
                break
        return LLMResponse(content=last_human, raw={"mock": True}, usage={"tokens": 0})


class OpenAILLMClient(LLMClient):
    """OpenAI-compatible LangChain chat model adapter."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        llm_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        extra = {}
        if base_url:
            extra["base_url"] = base_url
        self.client = ChatOpenAI(
            model=llm_model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, **extra
        )

    def complete(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        response = self.client.invoke(messages, **kwargs)
        text = getattr(response, "content", "") or str(response)
        usage = getattr(response, "response_metadata", {})
        return LLMResponse(content=text, raw=response, usage=usage)


def build_llm_client(mode: str = "openai", **kwargs) -> LLMClient:
    if mode == "mock":
        return MockLLMClient()
    return OpenAILLMClient(**kwargs)
