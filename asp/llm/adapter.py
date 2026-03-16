"""
LLMAdapter -- model-agnostic interface for LLM completion.

The adapter receives ONLY SanitizedContext (never raw prompts).
This is the final stage of the ASP pipeline.

Dependency Inversion: the pipeline depends on this protocol,
not on OpenAI/Llama/etc. directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from asp.types import SanitizedContext


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM adapter."""
    content: str
    model: str
    usage: dict[str, int]   # {"prompt_tokens": N, "completion_tokens": M}
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class LLMAdapter(Protocol):
    """Protocol for model-agnostic LLM completion."""

    async def complete(self, context: SanitizedContext) -> LLMResponse:
        """Send sanitized context to the LLM and get a response.

        The adapter is responsible for formatting the SanitizedContext
        into the model's expected input format (chat messages, prompt
        string, etc.).

        INVARIANT: This method MUST NOT access raw user prompts.
        It receives only the defense-processed SanitizedContext.
        """
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...
