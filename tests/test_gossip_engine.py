"""Tests for the gossip protocol engine."""

from __future__ import annotations

import numpy as np
import pytest

from asp.config import ASPConfig
from asp.encoder.attack_signature_db import AttackSignatureDB
from asp.gossip.engine import GossipEngine
from asp.gossip.peer import Peer
from asp.gossip.transport import InMemoryTransport
from asp.gossip.vaccine import vaccine_from_wire, vaccine_to_wire
from asp.types import Vaccine


@pytest.fixture
def gossip_config() -> ASPConfig:
    return ASPConfig(
        gossip_fanout=2,
        gossip_interval_s=0.1,
        gossip_max_rounds=5,
    )


class TestVaccineSerialization:

    def test_round_trip(self):
        embedding = np.random.default_rng(42).standard_normal(64)
        vaccine = Vaccine(
            signature_hash=Vaccine.compute_hash(embedding),
            attack_embedding=embedding,
            defense_module="adversarial_roleplay",
            discovered_by="node-0",
            metadata={"severity": "high"},
        )
        wire = vaccine_to_wire(vaccine)
        recovered = vaccine_from_wire(wire)

        assert recovered.signature_hash == vaccine.signature_hash
        assert recovered.defense_module == vaccine.defense_module
        assert np.allclose(recovered.attack_embedding, vaccine.attack_embedding)


class TestGossipEngine:

    def test_publish_new_vaccine(self, gossip_config):
        transport = InMemoryTransport()
        attack_db = AttackSignatureDB(dim=64)
        engine = GossipEngine("node-0", transport, attack_db, gossip_config)

        embedding = np.random.default_rng(99).standard_normal(64)
        vaccine = Vaccine(
            signature_hash=Vaccine.compute_hash(embedding),
            attack_embedding=embedding,
            defense_module="test",
            discovered_by="node-0",
        )

        assert engine.publish_vaccine(vaccine) is True
        assert attack_db.size == 1

    def test_duplicate_vaccine_rejected(self, gossip_config):
        transport = InMemoryTransport()
        attack_db = AttackSignatureDB(dim=64)
        engine = GossipEngine("node-0", transport, attack_db, gossip_config)

        embedding = np.random.default_rng(99).standard_normal(64)
        vaccine = Vaccine(
            signature_hash=Vaccine.compute_hash(embedding),
            attack_embedding=embedding,
            defense_module="test",
            discovered_by="node-0",
        )

        assert engine.publish_vaccine(vaccine) is True
        assert engine.publish_vaccine(vaccine) is False  # duplicate
        assert attack_db.size == 1  # still just one

    def test_peer_management(self, gossip_config):
        transport = InMemoryTransport()
        attack_db = AttackSignatureDB(dim=64)
        engine = GossipEngine("node-0", transport, attack_db, gossip_config)

        peer = Peer(peer_id="node-1", address="localhost:8001")
        engine.add_peer(peer)
        engine.remove_peer("node-1")
        # Should not crash on double remove
        engine.remove_peer("node-1")


class TestPeer:

    def test_seen_tracking(self):
        peer = Peer(peer_id="node-1", address="localhost:8001")
        assert not peer.has_seen("hash-1")
        peer.mark_seen("hash-1")
        assert peer.has_seen("hash-1")
