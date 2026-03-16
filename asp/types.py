"""
ASP Shared Domain Types

All value objects and enums used across module boundaries.
Kept in a single file to avoid circular imports and make
the domain language explicit.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ThreatLevel(Enum):
    """Graduated threat classification based on geometric distance
    from known attack manifolds in latent space."""
    BENIGN = auto()
    MONITOR = auto()      # near boundary -- log but allow
    WARN = auto()          # close to manifold -- augment context
    BLOCK = auto()         # inside manifold -- reject
    QUARANTINE = auto()    # novel attack shape -- hold for analysis


class Verdict(Enum):
    """Lifecycle of a defense decision through threshold validation.

    Ref: Thetacrypt threshold crypto -- a verdict is "untrusted" until
    N-of-M validator nodes co-sign it.
    """
    UNTRUSTED = auto()
    PENDING_VALIDATION = auto()
    VERIFIED_IMMUNITY = auto()
    REJECTED = auto()


class DefenseAction(Enum):
    """What a defense module did to mitigate the threat."""
    PASS_THROUGH = auto()       # benign, no modification
    CONTEXT_AUGMENT = auto()    # injected alignment anchors
    ROLEPLAY_REDIRECT = auto()  # redirected adversarial persona
    SANITIZE_AND_REWRITE = auto()
    FULL_BLOCK = auto()


# ---------------------------------------------------------------------------
# Core data objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThreatVector:
    """The geometric representation of a prompt's intent in latent space.

    This is the output of the Morphological Intent Encoder.  It contains
    the raw embedding plus derived similarity scores against the attack
    signature database.
    """
    embedding: NDArray[np.float64]          # shape: (dim,)
    max_attack_similarity: float            # cosine sim to nearest attack sig
    nearest_attack_id: str                  # id of the nearest signature
    threat_level: ThreatLevel
    timestamp: float = field(default_factory=time.time)

    def to_list(self) -> list[float]:
        return self.embedding.tolist()


@dataclass(frozen=True)
class SanitizedContext:
    """Output of the TEE sanitizer.  This is the ONLY thing the LLM sees.

    Invariant: no field in this object contains the raw user prompt.
    The `rewritten_prompt` is a defense-augmented version.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rewritten_prompt: str = ""
    alignment_preamble: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MitigationPayload:
    """The result of a defense module's mitigation action."""
    defense_module: str          # which module handled it
    action: DefenseAction
    sanitized_context: SanitizedContext
    explanation: str = ""        # human-readable rationale (for telemetry)


@dataclass(frozen=True)
class ThresholdSignatureBlock:
    """Aggregated N-of-M threshold signature over a verdict.

    Ref: Thetacrypt -- each `shares` entry is a partial signature
    from one validator node.  The block is valid when len(shares) >= threshold.
    """
    verdict: Verdict
    request_id: str
    threshold: int                         # N (minimum required)
    total_nodes: int                       # M (total validator set)
    shares: tuple[SignatureShare, ...] = ()
    aggregated_signature: bytes = b""      # reconstructed from shares

    @property
    def is_valid(self) -> bool:
        return len(self.shares) >= self.threshold and self.aggregated_signature != b""


@dataclass(frozen=True)
class SignatureShare:
    """A single node's partial signature contribution."""
    node_id: str
    share_index: int
    share_value: bytes
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Vaccine:
    """A new defense signature discovered by one node, propagated to all.

    Idempotent: deduplicated by `signature_hash` across the gossip network.
    """
    signature_hash: str                    # SHA-256 of the attack embedding
    attack_embedding: NDArray[np.float64]  # the new vector to add to attack DB
    defense_module: str                    # which module can handle this class
    discovered_by: str                     # node ID that found it
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def compute_hash(embedding: NDArray[np.float64]) -> str:
        return hashlib.sha256(embedding.tobytes()).hexdigest()
