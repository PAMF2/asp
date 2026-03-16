"""
ValidatorNode -- a single node in the threshold validator set.

Each node has a unique identity, a secret share, and can sign
verdicts.  Nodes are the "voters" in the N-of-M threshold scheme.

Ref: Thetacrypt -- validator nodes form a "quorum of trust".
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from asp.threshold.share import sign_verdict
from asp.types import SignatureShare, ThreatVector, Verdict


@dataclass
class ValidatorNode:
    """A single threshold validator node."""

    node_id: str
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    share_index: int = 0  # assigned by registry

    def sign(
        self, request_id: str, threat: ThreatVector, verdict: Verdict
    ) -> SignatureShare:
        """Produce this node's partial signature for a verdict.

        The node independently evaluates whether the verdict is
        consistent with the threat vector.  In production, nodes
        would run their own encoder and cross-validate.  For
        hackathon, they sign if the verdict matches their local
        threat assessment.
        """
        signature = sign_verdict(
            self.secret, request_id, verdict.name
        )

        return SignatureShare(
            node_id=self.node_id,
            share_index=self.share_index,
            share_value=signature,
            timestamp=time.time(),
        )

    def evaluate_locally(self, threat: ThreatVector) -> Verdict:
        """Local threat assessment to validate against proposed verdict.

        Simplified for hackathon: derive verdict from threat level.
        Production: each node runs its own encoder.
        """
        from asp.types import ThreatLevel

        if threat.threat_level in (ThreatLevel.BLOCK, ThreatLevel.QUARANTINE):
            return Verdict.REJECTED
        return Verdict.VERIFIED_IMMUNITY
