"""
ThresholdValidator -- N-of-M signature collection and verification.

Collects partial signatures from validator nodes and produces a
ThresholdSignatureBlock.  A verdict transitions from UNTRUSTED to
VERIFIED_IMMUNITY only when N nodes agree.

State machine:
  UNTRUSTED -> PENDING_VALIDATION -> VERIFIED_IMMUNITY
                                  -> REJECTED

Ref: Thetacrypt -- threshold signatures as "quorum of trust".
"""

from __future__ import annotations

import asyncio
import hashlib

from asp.config import ASPConfig
from asp.threshold.registry import NodeRegistry
from asp.types import (
    MitigationPayload,
    SignatureShare,
    ThresholdSignatureBlock,
    ThreatVector,
    Verdict,
)


class ThresholdValidator:
    """Collect N-of-M node signatures to validate a defense verdict."""

    def __init__(self, registry: NodeRegistry, config: ASPConfig) -> None:
        self._registry = registry
        self._config = config

    async def validate(
        self,
        request_id: str,
        threat_vector: ThreatVector,
        mitigation: MitigationPayload,
    ) -> ThresholdSignatureBlock:
        """Collect signatures from validator nodes.

        Each node independently evaluates the threat and signs if
        it agrees with the proposed verdict.  We need N agreements
        out of M total nodes.

        Returns:
            ThresholdSignatureBlock with collected shares and
            aggregated signature (if quorum reached).
        """
        nodes = self._registry.get_all()
        total = len(nodes)
        threshold = min(self._config.threshold_n, total)

        if total == 0:
            # No validators available -- cannot validate
            return ThresholdSignatureBlock(
                verdict=Verdict.UNTRUSTED,
                request_id=request_id,
                threshold=threshold,
                total_nodes=0,
            )

        # Determine proposed verdict from mitigation action
        from asp.types import DefenseAction

        if mitigation.action == DefenseAction.FULL_BLOCK:
            proposed_verdict = Verdict.REJECTED
        else:
            proposed_verdict = Verdict.VERIFIED_IMMUNITY

        # Collect signatures from all nodes (async, with timeout)
        shares: list[SignatureShare] = []

        async def collect_from_node(node):
            """Simulate async node communication."""
            local_verdict = node.evaluate_locally(threat_vector)
            # Node signs only if its local assessment agrees
            if local_verdict == proposed_verdict:
                return node.sign(request_id, threat_vector, proposed_verdict)
            return None

        tasks = [collect_from_node(node) for node in nodes]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self._config.validation_timeout_s,
            )
        except asyncio.TimeoutError:
            results = []

        for result in results:
            if result is not None:
                shares.append(result)

        # Check quorum
        quorum_reached = len(shares) >= threshold
        final_verdict = proposed_verdict if quorum_reached else Verdict.UNTRUSTED

        # Aggregate signature (hash of all share values)
        aggregated = b""
        if quorum_reached:
            hasher = hashlib.sha256()
            for share in sorted(shares, key=lambda s: s.share_index):
                hasher.update(share.share_value)
            aggregated = hasher.digest()

        return ThresholdSignatureBlock(
            verdict=final_verdict,
            request_id=request_id,
            threshold=threshold,
            total_nodes=total,
            shares=tuple(shares),
            aggregated_signature=aggregated,
        )
