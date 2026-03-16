"""
GossipEngine -- epidemic broadcast for vaccine propagation.

Implements rumor-spreading (push) gossip:
1. Node discovers a new attack signature (vaccine)
2. Node selects `fanout` random peers from its peer list
3. Node sends vaccine to selected peers
4. Each receiving peer that hasn't seen this vaccine repeats step 2-3
5. Convergence in O(log N) rounds for N nodes

Vaccines are idempotent: deduplicated by signature_hash.

Cite: epidemic algorithms for replicated database maintenance
(Demers et al., 1987).  The specific model here is "rumor mongering"
where a node stops pushing a vaccine after it encounters peers that
already have it (with probability-based stopping).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable

from asp.config import ASPConfig
from asp.encoder.attack_signature_db import AttackSignature, AttackSignatureDB
from asp.gossip.peer import Peer
from asp.gossip.transport import GossipTransport
from asp.gossip.vaccine import vaccine_from_wire, vaccine_to_wire
from asp.types import Vaccine

logger = logging.getLogger(__name__)


class GossipEngine:
    """Async epidemic gossip for vaccine propagation.

    Lifecycle:
    1. publish_vaccine() -- inject a new vaccine into the gossip network
    2. _gossip_round() -- periodically push unseen vaccines to peers
    3. handle_received() -- process incoming vaccines from peers

    The engine runs as a background asyncio task.
    """

    def __init__(
        self,
        node_id: str,
        transport: GossipTransport,
        attack_db: AttackSignatureDB,
        config: ASPConfig,
        on_new_vaccine: Callable[[Vaccine], None] | None = None,
    ) -> None:
        self._node_id = node_id
        self._transport = transport
        self._attack_db = attack_db
        self._config = config
        self._on_new_vaccine = on_new_vaccine

        self._peers: dict[str, Peer] = {}
        self._seen: set[str] = set()         # vaccine hashes we've seen
        self._pending: list[Vaccine] = []    # vaccines to gossip
        self._running = False
        self._task: asyncio.Task | None = None

    def add_peer(self, peer: Peer) -> None:
        self._peers[peer.peer_id] = peer

    def remove_peer(self, peer_id: str) -> None:
        self._peers.pop(peer_id, None)

    async def start(self) -> None:
        """Start the background gossip loop."""
        self._running = True
        self._task = asyncio.create_task(self._gossip_loop())
        logger.info("Gossip engine started for node %s", self._node_id)

    async def stop(self) -> None:
        """Stop the gossip loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Gossip engine stopped for node %s", self._node_id)

    def publish_vaccine(self, vaccine: Vaccine) -> bool:
        """Inject a new vaccine for propagation.

        Returns True if the vaccine is new, False if already seen.
        """
        if vaccine.signature_hash in self._seen:
            return False

        self._seen.add(vaccine.signature_hash)
        self._pending.append(vaccine)

        # Also add to local attack DB
        self._attack_db.add(
            AttackSignature(
                signature_id=vaccine.signature_hash,
                embedding=vaccine.attack_embedding,
                category=vaccine.defense_module,
                source=vaccine.discovered_by,
            )
        )

        if self._on_new_vaccine:
            self._on_new_vaccine(vaccine)

        logger.info(
            "Vaccine %s published by node %s",
            vaccine.signature_hash[:16],
            self._node_id,
        )
        return True

    async def handle_received(self, peer_id: str, message: str) -> None:
        """Process a vaccine received from a peer."""
        try:
            vaccine = vaccine_from_wire(message)
        except Exception:
            logger.warning("Invalid vaccine message from %s", peer_id)
            return

        if vaccine.signature_hash in self._seen:
            # Already have this vaccine -- update peer's seen set
            peer = self._peers.get(peer_id)
            if peer:
                peer.mark_seen(vaccine.signature_hash)
            return

        # New vaccine -- accept and re-gossip
        self.publish_vaccine(vaccine)
        logger.info(
            "Vaccine %s received from peer %s, will propagate",
            vaccine.signature_hash[:16],
            peer_id,
        )

    async def _gossip_loop(self) -> None:
        """Background loop: push pending vaccines to random peers."""
        while self._running:
            await asyncio.sleep(self._config.gossip_interval_s)
            await self._gossip_round()

    async def _gossip_round(self) -> None:
        """One round of epidemic gossip."""
        if not self._pending or not self._peers:
            return

        # Select random peers (fanout)
        available = [p for p in self._peers.values() if p.is_healthy]
        if not available:
            return

        fanout = min(self._config.gossip_fanout, len(available))
        targets = random.sample(available, fanout)

        vaccines_to_send = list(self._pending)
        self._pending.clear()

        for vaccine in vaccines_to_send:
            wire = vaccine_to_wire(vaccine)
            any_success = False
            for peer in targets:
                if not peer.has_seen(vaccine.signature_hash):
                    success = await self._transport.send(peer, wire)
                    if success:
                        any_success = True
                        peer.mark_seen(vaccine.signature_hash)
                        peer.last_contact = time.time()
            # Re-queue vaccine if no peer received it
            if not any_success:
                self._pending.append(vaccine)
