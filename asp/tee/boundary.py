"""
TEEBoundary -- the orchestrator inside the trusted enclave.

This is the entry point for all prompt processing.  It wires
together: sanitizer -> encoder -> threshold validator -> defense router.

The boundary enforces the invariant that raw prompts never leave
the enclave.  After processing, only SanitizedContext (with the
defense module's rewritten prompt) exits.

Ref: NDAI Agreements TEE paper -- the TEE boundary is the
"data handling policy" enforcement point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from asp.types import (
    MitigationPayload,
    SanitizedContext,
    ThresholdSignatureBlock,
    ThreatVector,
    Verdict,
)

if TYPE_CHECKING:
    from asp.defense.router import DefenseRouter
    from asp.encoder.intent_encoder import MorphologicalIntentEncoder
    from asp.tee.attestation import DstackAttestation
    from asp.tee.sanitizer import SanitizerGateway
    from asp.threshold.validator import ThresholdValidator


@dataclass(frozen=True)
class PipelineResult:
    """Complete result of processing a prompt through the TEE pipeline."""
    sanitized_context: SanitizedContext
    threat_vector: ThreatVector
    mitigation: MitigationPayload
    signature_block: ThresholdSignatureBlock
    verdict: Verdict


class TEEBoundary:
    """Orchestrates the full defense pipeline inside the TEE.

    Lifecycle of a request:
    1. Sanitizer strips raw prompt -> SanitizedContext skeleton
    2. Encoder projects raw prompt -> ThreatVector (latent space)
    3. Defense router selects module -> MitigationPayload
    4. Threshold validator collects N-of-M signatures -> verdict
    5. If VERIFIED_IMMUNITY: release sanitized context to LLM
       If REJECTED: return error, nothing leaves the enclave

    CRITICAL: raw_prompt is a local variable.  It is NEVER stored
    as instance state, NEVER logged, NEVER returned.
    """

    def __init__(
        self,
        sanitizer: SanitizerGateway,
        encoder: MorphologicalIntentEncoder,
        defense_router: DefenseRouter,
        threshold_validator: ThresholdValidator,
        attestation: DstackAttestation | None = None,
    ) -> None:
        self._sanitizer = sanitizer
        self._encoder = encoder
        self._defense_router = defense_router
        self._threshold_validator = threshold_validator
        self._attestation = attestation

    async def process(self, raw_prompt: str) -> PipelineResult:
        """Process a raw prompt through the full ASP pipeline.

        Args:
            raw_prompt: The untrusted user input.  Consumed and destroyed
                        within this method.  Never stored.

        Returns:
            PipelineResult with sanitized context safe for LLM consumption.

        Raises:
            SecurityError: if threshold validation rejects the request.
        """
        try:
            # Step 1: Sanitize (skeleton only -- no raw content in output)
            sanitized = self._sanitizer.sanitize(raw_prompt)

            # Step 2: Encode raw prompt into latent space
            # This is the LAST time raw_prompt is accessed.
            threat_vector = self._encoder.encode(raw_prompt)
        finally:
            # raw_prompt is now dead.  Python GC will collect it.
            # In a real TEE, the enclave memory page is zeroed.
            del raw_prompt

        # Step 3: Route to defense module and get mitigation
        mitigation = self._defense_router.route(threat_vector, sanitized)

        # Step 4: Threshold validation
        signature_block = await self._threshold_validator.validate(
            request_id=sanitized.request_id,
            threat_vector=threat_vector,
            mitigation=mitigation,
        )

        verdict = signature_block.verdict

        if verdict == Verdict.REJECTED:
            raise SecurityError(
                f"Request {sanitized.request_id} rejected by threshold consensus"
            )

        return PipelineResult(
            sanitized_context=mitigation.sanitized_context,
            threat_vector=threat_vector,
            mitigation=mitigation,
            signature_block=signature_block,
            verdict=verdict,
        )


class SecurityError(Exception):
    """Raised when threshold validation rejects a request."""
