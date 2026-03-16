"""
Telemetry Payload Schemas

Dataclass definitions for the three required JSON-RPC 2.0 payloads:
1. threat_signature_vector
2. mitigation_payload
3. threshold_signature_block
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ThreatSignatureVectorPayload:
    """JSON-RPC payload for reporting a threat signature."""
    request_id: str
    embedding: list[float]          # serialized numpy array
    max_attack_similarity: float
    nearest_attack_id: str
    threat_level: str               # enum name
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MitigationPayloadTelemetry:
    """JSON-RPC payload for reporting a mitigation action."""
    request_id: str
    defense_module: str
    action: str                     # DefenseAction enum name
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSignatureBlockPayload:
    """JSON-RPC payload for reporting threshold validation result."""
    request_id: str
    verdict: str                    # Verdict enum name
    threshold: int
    total_nodes: int
    shares_collected: int
    is_valid: bool
    aggregated_signature_hex: str   # hex-encoded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
