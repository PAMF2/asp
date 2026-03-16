"""Tests for threshold validation."""

from __future__ import annotations

import pytest

from asp.threshold.registry import NodeRegistry
from asp.threshold.validator import ThresholdValidator
from asp.threshold.node import ValidatorNode
from asp.threshold.share import split_secret, reconstruct_secret
from asp.types import (
    DefenseAction,
    MitigationPayload,
    SanitizedContext,
    ThreatLevel,
    ThreatVector,
    Verdict,
)
import numpy as np


class TestShamirSecretSharing:

    def test_split_and_reconstruct(self):
        secret = b"this is a 32-byte secret value!!"
        shares = split_secret(secret, n=3, m=5)
        assert len(shares) == 5

        recovered = reconstruct_secret(shares[:3], n=3)
        assert recovered == secret

    def test_insufficient_shares_raises(self):
        secret = b"test secret value here 32 bytes!"
        shares = split_secret(secret, n=3, m=5)
        with pytest.raises(ValueError, match="Need at least"):
            reconstruct_secret(shares[:2], n=3)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="Invalid threshold"):
            split_secret(b"secret", n=0, m=3)

    def test_n_equals_m(self):
        secret = b"all shares needed 32 bytes val!!"
        shares = split_secret(secret, n=3, m=3)
        recovered = reconstruct_secret(shares, n=3)
        assert recovered == secret


@pytest.mark.asyncio
class TestThresholdValidator:

    async def test_benign_prompt_gets_verified(self, threshold_validator, config):
        threat = ThreatVector(
            embedding=np.zeros(64),
            max_attack_similarity=0.1,
            nearest_attack_id="",
            threat_level=ThreatLevel.BENIGN,
        )
        mitigation = MitigationPayload(
            defense_module="test",
            action=DefenseAction.PASS_THROUGH,
            sanitized_context=SanitizedContext(request_id="req-1"),
        )
        block = await threshold_validator.validate("req-1", threat, mitigation)
        assert block.verdict == Verdict.VERIFIED_IMMUNITY
        assert block.is_valid

    async def test_no_nodes_returns_untrusted(self, config):
        empty_registry = NodeRegistry()
        validator = ThresholdValidator(registry=empty_registry, config=config)

        threat = ThreatVector(
            embedding=np.zeros(64),
            max_attack_similarity=0.0,
            nearest_attack_id="",
            threat_level=ThreatLevel.BENIGN,
        )
        mitigation = MitigationPayload(
            defense_module="test",
            action=DefenseAction.PASS_THROUGH,
            sanitized_context=SanitizedContext(request_id="req-2"),
        )
        block = await validator.validate("req-2", threat, mitigation)
        assert block.verdict == Verdict.UNTRUSTED
