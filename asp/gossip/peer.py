"""
Peer -- represents a federated ASL node in the gossip network.

Each peer has an address and a set of vaccine hashes it has already
seen (for deduplication).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Peer:
    """A node in the federated ASL gossip network."""
    peer_id: str
    address: str                           # host:port or URL
    seen_vaccines: set[str] = field(default_factory=set)
    is_healthy: bool = True
    last_contact: float = 0.0

    def has_seen(self, vaccine_hash: str) -> bool:
        return vaccine_hash in self.seen_vaccines

    def mark_seen(self, vaccine_hash: str) -> None:
        self.seen_vaccines.add(vaccine_hash)
