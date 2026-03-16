"""Tests for defense modules and router."""

from __future__ import annotations

import numpy as np
import pytest

from asp.defense.adversarial_roleplay import AdversarialRoleplayModule
from asp.defense.context_injection import ContextInjectionModule
from asp.defense.router import DefenseRouter
from asp.types import (
    DefenseAction,
    SanitizedContext,
    ThreatLevel,
    ThreatVector,
)


class TestDefenseRouter:

    def test_benign_passes_through(self, defense_router):
        threat = ThreatVector(
            embedding=np.zeros(64),
            max_attack_similarity=0.1,
            nearest_attack_id="",
            threat_level=ThreatLevel.BENIGN,
        )
        ctx = SanitizedContext(request_id="req-1")
        result = defense_router.route(threat, ctx)
        assert result.action == DefenseAction.PASS_THROUGH

    def test_no_modules_fallback_blocks(self):
        empty_router = DefenseRouter()
        threat = ThreatVector(
            embedding=np.zeros(64),
            max_attack_similarity=0.9,
            nearest_attack_id="atk-1",
            threat_level=ThreatLevel.BLOCK,
        )
        ctx = SanitizedContext(request_id="req-2")
        result = empty_router.route(threat, ctx)
        assert result.action == DefenseAction.FULL_BLOCK
        assert result.defense_module == "router_fallback"


class TestAdversarialRoleplayModule:

    def test_has_capability_vector(self, embedding_adapter):
        module = AdversarialRoleplayModule(embedding_adapter)
        assert module.capability_vector.shape == (64,)
        # Should be normalized
        assert abs(np.linalg.norm(module.capability_vector) - 1.0) < 1e-6

    def test_blocks_high_threat(self, embedding_adapter):
        module = AdversarialRoleplayModule(embedding_adapter)
        threat = ThreatVector(
            embedding=module.capability_vector,  # perfect match
            max_attack_similarity=0.95,
            nearest_attack_id="atk-1",
            threat_level=ThreatLevel.BLOCK,
        )
        ctx = SanitizedContext(request_id="req-3")
        result = module.mitigate(threat, ctx)
        assert result.action == DefenseAction.FULL_BLOCK

    def test_redirects_warn_level(self, embedding_adapter):
        module = AdversarialRoleplayModule(embedding_adapter)
        threat = ThreatVector(
            embedding=module.capability_vector,
            max_attack_similarity=0.75,
            nearest_attack_id="atk-1",
            threat_level=ThreatLevel.WARN,
        )
        ctx = SanitizedContext(request_id="req-4")
        result = module.mitigate(threat, ctx)
        assert result.action == DefenseAction.ROLEPLAY_REDIRECT


class TestContextInjectionModule:

    def test_augments_warn_level(self, embedding_adapter):
        module = ContextInjectionModule(embedding_adapter)
        threat = ThreatVector(
            embedding=module.capability_vector,
            max_attack_similarity=0.75,
            nearest_attack_id="atk-1",
            threat_level=ThreatLevel.WARN,
        )
        ctx = SanitizedContext(request_id="req-5")
        result = module.mitigate(threat, ctx)
        assert result.action == DefenseAction.CONTEXT_AUGMENT
        assert "SYSTEM BOUNDARY" in result.sanitized_context.rewritten_prompt
