"""
End-to-end pipeline integration test.

Verifies the full flow: raw prompt -> TEE boundary -> sanitized output.
Confirms that raw prompts never appear in any output.
"""

from __future__ import annotations

import pytest

from asp.tee.boundary import TEEBoundary, SecurityError
from asp.types import ThreatLevel, Verdict


@pytest.fixture
def boundary(sanitizer, encoder, defense_router, threshold_validator):
    return TEEBoundary(
        sanitizer=sanitizer,
        encoder=encoder,
        defense_router=defense_router,
        threshold_validator=threshold_validator,
    )


@pytest.mark.asyncio
class TestE2EPipeline:
    """Full pipeline integration tests."""

    async def test_benign_prompt_passes_through(self, boundary):
        """A normal prompt should pass through with VERIFIED_IMMUNITY."""
        result = await boundary.process("What is the weather today?")

        assert result.verdict == Verdict.VERIFIED_IMMUNITY
        assert result.sanitized_context.rewritten_prompt != ""
        assert result.signature_block.is_valid

    async def test_raw_prompt_never_in_output(self, boundary):
        """The raw prompt text must NEVER appear in any output field."""
        raw = "This is a unique test string 8f3k2j5"
        result = await boundary.process(raw)

        # Check every string field in the output
        assert raw not in result.sanitized_context.rewritten_prompt
        assert raw not in result.sanitized_context.alignment_preamble
        assert raw not in result.mitigation.explanation
        assert raw not in str(result.sanitized_context.metadata)

    async def test_threat_vector_has_embedding(self, boundary):
        """Threat vector should contain a valid embedding."""
        result = await boundary.process("Tell me a joke")

        assert result.threat_vector.embedding.shape == (64,)
        assert result.threat_vector.threat_level is not None

    async def test_threshold_block_has_sufficient_shares(self, boundary):
        """Signature block should have at least N shares."""
        result = await boundary.process("Hello world")

        assert len(result.signature_block.shares) >= result.signature_block.threshold
        assert result.signature_block.aggregated_signature != b""

    async def test_sanitized_context_has_preamble(self, boundary):
        """Every output must include the alignment preamble."""
        result = await boundary.process("Explain quantum computing")

        assert "Alignment Security Protocol" in result.sanitized_context.alignment_preamble
