"""
Vaccine serialization helpers.

Vaccines are the unit of propagation in the gossip network.
This module handles conversion to/from wire format.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from asp.types import Vaccine


def vaccine_to_wire(vaccine: Vaccine) -> str:
    """Serialize a Vaccine to JSON wire format."""
    return json.dumps({
        "signature_hash": vaccine.signature_hash,
        "attack_embedding": vaccine.attack_embedding.tolist(),
        "defense_module": vaccine.defense_module,
        "discovered_by": vaccine.discovered_by,
        "timestamp": vaccine.timestamp,
        "metadata": vaccine.metadata,
    })


def vaccine_from_wire(data: str) -> Vaccine:
    """Deserialize a Vaccine from JSON wire format.

    Validates that signature_hash matches the embedding to prevent
    poisoning attacks from malicious peers.
    """
    obj = json.loads(data)
    embedding = np.array(obj["attack_embedding"], dtype=np.float64)

    # Verify integrity: hash must match the actual embedding
    import hashlib
    expected_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
    received_hash = obj["signature_hash"]
    if received_hash != expected_hash:
        raise ValueError(
            f"Vaccine signature_hash mismatch: received {received_hash[:16]}... "
            f"expected {expected_hash[:16]}..."
        )

    return Vaccine(
        signature_hash=received_hash,
        attack_embedding=embedding,
        defense_module=obj["defense_module"],
        discovered_by=obj["discovered_by"],
        timestamp=obj.get("timestamp", 0.0),
        metadata=obj.get("metadata", {}),
    )
