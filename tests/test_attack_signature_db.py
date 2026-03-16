"""Tests for the attack signature database."""

from __future__ import annotations

import numpy as np

from asp.encoder.attack_signature_db import AttackSignature, AttackSignatureDB


class TestAttackSignatureDB:

    def test_add_and_search(self):
        db = AttackSignatureDB(dim=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        db.add(AttackSignature(signature_id="s1", embedding=vec))

        results = db.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "s1"
        assert results[0][1] > 0.99  # near-perfect match

    def test_duplicate_rejected(self):
        db = AttackSignatureDB(dim=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        sig = AttackSignature(signature_id="s1", embedding=vec)
        assert db.add(sig) is True
        assert db.add(sig) is False
        assert db.size == 1

    def test_empty_db_search(self):
        db = AttackSignatureDB(dim=4)
        results = db.search(np.array([1.0, 0.0, 0.0, 0.0]))
        assert results == []

    def test_batch_add(self):
        db = AttackSignatureDB(dim=4)
        sigs = [
            AttackSignature(f"s{i}", np.random.default_rng(i).standard_normal(4))
            for i in range(5)
        ]
        added = db.add_batch(sigs)
        assert added == 5
        assert db.size == 5

    def test_contains(self):
        db = AttackSignatureDB(dim=4)
        db.add(AttackSignature("s1", np.zeros(4)))
        assert db.contains("s1")
        assert not db.contains("s2")
