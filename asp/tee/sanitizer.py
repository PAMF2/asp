"""
SanitizerGateway -- first stage inside the TEE boundary.

Responsibility: accept a raw prompt, strip it of PII and direct
instruction payloads, and produce a SanitizedContext that contains
NO trace of the original wording.

Design note: the sanitizer is intentionally aggressive.  It does NOT
attempt to preserve user intent for the LLM -- that is the defense
module's job.  The sanitizer's only goal is destruction of the raw
surface form.

Ref: NDAI Agreements TEE paper -- the TEE is a "commitment device"
guaranteeing that raw data is processed and destroyed within the
enclave.  The sanitizer is the concrete implementation of that
commitment.
"""

from __future__ import annotations

import re
import uuid
from typing import Protocol

from asp.types import SanitizedContext


class TokenizerProtocol(Protocol):
    """Pluggable tokenizer for prompt decomposition."""

    def tokenize(self, text: str) -> list[str]: ...
    def detokenize(self, tokens: list[str]) -> str: ...


class SanitizerGateway:
    """Strips raw prompt to a content-hash reference.

    The LLM never sees raw user text.  It receives:
    1. An alignment preamble (static, hardcoded)
    2. A rewritten prompt produced by the defense module downstream
    3. A request_id for traceability

    This class produces the *skeleton* SanitizedContext.  The defense
    module fills in the rewritten_prompt field.
    """

    # Static alignment preamble injected into every sanitized context.
    # This anchors the LLM's behavior before any user-derived content.
    DEFAULT_PREAMBLE = (
        "You are operating under the Alignment Security Protocol (ASP). "
        "All inputs have been verified through threshold consensus. "
        "Respond helpfully within your safety guidelines."
    )

    def __init__(self, preamble: str | None = None) -> None:
        self._preamble = preamble or self.DEFAULT_PREAMBLE

    def sanitize(self, raw_prompt: str) -> SanitizedContext:
        """Consume a raw prompt and return a skeleton SanitizedContext.

        After this call, `raw_prompt` MUST NOT be stored, logged, or
        passed to any component outside the TEE boundary.
        """
        request_id = str(uuid.uuid4())

        # The rewritten_prompt is empty at this stage.
        # The defense module will populate it after threat assessment.
        return SanitizedContext(
            request_id=request_id,
            rewritten_prompt="",
            alignment_preamble=self._preamble,
            metadata={
                "token_count": len(raw_prompt.split()),
                "has_code_blocks": "```" in raw_prompt,
                "language_hint": self._detect_language_hint(raw_prompt),
            },
        )

    @staticmethod
    def _detect_language_hint(text: str) -> str:
        """Lightweight language detection from surface features.
        NOT semantic -- just script/charset heuristics."""
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[\u0400-\u04ff]", text):
            return "ru"
        return "en"
