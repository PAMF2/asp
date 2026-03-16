"""
Llama/Local LLM Adapter

Formats SanitizedContext for local model inference (e.g., llama.cpp,
Ollama, vLLM).
"""

from __future__ import annotations

from asp.config import ASPConfig
from asp.llm.adapter import LLMResponse
from asp.types import SanitizedContext


class LlamaAdapter:
    """Local Llama model adapter.

    Supports any OpenAI-compatible local server (Ollama, vLLM, etc.)
    by targeting their /v1/chat/completions endpoint.
    """

    def __init__(self, config: ASPConfig, base_url: str = "http://localhost:11434") -> None:
        self._config = config
        self._base_url = base_url
        self._model = config.llm_model

    @property
    def model_name(self) -> str:
        return f"local:{self._model}"

    async def complete(self, context: SanitizedContext) -> LLMResponse:
        """Send sanitized context to local Llama server."""
        # Placeholder for actual HTTP call to local server
        # import aiohttp
        # async with aiohttp.ClientSession() as session:
        #     resp = await session.post(
        #         f"{self._base_url}/v1/chat/completions",
        #         json={"model": self._model, "messages": messages},
        #     )
        #     data = await resp.json()

        return LLMResponse(
            content="[Stub] Local LLM response placeholder",
            model=self._model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )
