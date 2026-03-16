"""
GossipTransport -- pluggable network layer for gossip protocol.

Protocol-based abstraction so the gossip engine does not depend
on a specific network implementation (HTTP, TCP, in-memory for tests).
"""

from __future__ import annotations

from typing import Protocol

from asp.gossip.peer import Peer


class GossipTransport(Protocol):
    """Protocol for sending gossip messages to peers."""

    async def send(self, peer: Peer, message: str) -> bool:
        """Send a message to a peer.  Returns True on success."""
        ...

    async def receive(self) -> tuple[str, str]:
        """Receive a message.  Returns (peer_id, message)."""
        ...


class InMemoryTransport:
    """In-memory transport for testing.  Messages are queued."""

    def __init__(self) -> None:
        self._queues: dict[str, list[tuple[str, str]]] = {}  # peer_id -> [(from, msg)]
        self._inbox: list[tuple[str, str]] = []

    def register_peer(self, peer_id: str) -> None:
        self._queues[peer_id] = []

    async def send(self, peer: Peer, message: str) -> bool:
        if peer.peer_id not in self._queues:
            return False
        self._queues[peer.peer_id].append(("local", message))
        return True

    async def receive(self) -> tuple[str, str]:
        if self._inbox:
            return self._inbox.pop(0)
        raise QueueEmpty("No messages")

    def inject_message(self, from_peer: str, message: str) -> None:
        """Test helper: inject a message into the inbox."""
        self._inbox.append((from_peer, message))


class QueueEmpty(Exception):
    """Raised when the transport inbox has no pending messages."""
