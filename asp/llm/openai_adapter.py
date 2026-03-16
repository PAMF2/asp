"""
OpenAI LLM Adapter

Formats SanitizedContext as OpenAI chat messages and calls the API.
"""

from __future__ import annotations

from asp.config import ASPConfig
from asp.llm.adapter import LLMResponse
from asp.types import SanitizedContext


class OpenAIAdapter:
    """OpenAI API adapter.

    In production, uses the openai Python SDK.
    For hackathon, provides the interface with a stub implementation
    that can be swapped in when API key is available.
    """

    def __init__(self, config: ASPConfig) -> None:
        self._config = config
        self._model = config.llm_model

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(self, context: SanitizedContext) -> LLMResponse:
        """Send sanitized context to OpenAI.

        Message format:
        - system: alignment_preamble
        - user: rewritten_prompt
        """
        messages = [
            {"role": "system", "content": context.alignment_preamble},
            {"role": "user", "content": context.rewritten_prompt},
        ]

        # Placeholder: actual OpenAI call
        # from openai import AsyncOpenAI
        # client = AsyncOpenAI(api_key=self._config.llm_api_key)
        # response = await client.chat.completions.create(
        #     model=self._model,
        #     messages=messages,
        # )
        # return LLMResponse(
        #     content=response.choices[0].message.content,
        #     model=response.model,
        #     usage={"prompt_tokens": response.usage.prompt_tokens,
        #            "completion_tokens": response.usage.completion_tokens},
        # )

        return LLMResponse(
            content="[Stub] LLM response placeholder",
            model=self._model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )
