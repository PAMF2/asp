"""
DefenseRouter -- selects the best defense module for a given threat.

Uses vector-space similarity between the threat vector and each
registered module's capability vector.  This is itself a geometric
operation -- no hardcoded routing rules.

Strategy Pattern: the router delegates to whichever module scores
highest on evaluate().  Ties broken by registration order.
"""

from __future__ import annotations

from asp.defense.base import ASPDefenseModule
from asp.types import (
    DefenseAction,
    MitigationPayload,
    SanitizedContext,
    ThreatLevel,
    ThreatVector,
)


class DefenseRouter:
    """Route threats to the most capable defense module."""

    def __init__(self) -> None:
        self._modules: list[ASPDefenseModule] = []

    def register(self, module: ASPDefenseModule) -> None:
        """Register a defense module.  Order matters for tie-breaking."""
        self._modules.append(module)

    def route(
        self, threat: ThreatVector, context: SanitizedContext
    ) -> MitigationPayload:
        """Select and invoke the best defense module.

        If the threat is BENIGN, return a pass-through (no modification).
        Otherwise, find the module with the highest evaluate() score
        that handles the threat level.
        """
        if threat.threat_level == ThreatLevel.BENIGN:
            return MitigationPayload(
                defense_module="passthrough",
                action=DefenseAction.PASS_THROUGH,
                sanitized_context=SanitizedContext(
                    request_id=context.request_id,
                    rewritten_prompt="Process the user's request normally.",
                    alignment_preamble=context.alignment_preamble,
                    metadata=context.metadata,
                ),
                explanation="Threat level BENIGN -- no defense needed.",
            )

        # Score all eligible modules
        candidates: list[tuple[float, ASPDefenseModule]] = []
        for module in self._modules:
            if module.handles_threat_level(threat.threat_level):
                score = module.evaluate(threat)
                candidates.append((score, module))

        if not candidates:
            # No module can handle this -- fail safe with full block
            return MitigationPayload(
                defense_module="router_fallback",
                action=DefenseAction.FULL_BLOCK,
                sanitized_context=SanitizedContext(
                    request_id=context.request_id,
                    rewritten_prompt=(
                        "A potential threat was detected but no defense module "
                        "is available. The request has been blocked as a precaution."
                    ),
                    alignment_preamble=context.alignment_preamble,
                    metadata={**context.metadata, "fallback_block": True},
                ),
                explanation="No defense module handles this threat level. Fail-safe block.",
            )

        # Select highest-scoring module
        candidates.sort(key=lambda x: x[0], reverse=True)
        _score, best_module = candidates[0]

        return best_module.mitigate(threat, context)
