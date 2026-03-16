"""
TEE Attestation Verification

Verifies that the code running inside the TEE enclave matches
the expected measurement (hash of the enclave binary).

Ref: NDAI Agreements TEE paper -- remote attestation proves to
external parties that the sanitizer code has not been tampered with.
Dstack provides the attestation API.

In hackathon/dev mode, attestation is stubbed (always passes).
In production, this would call the Dstack attestation endpoint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


class AttestationProvider(Protocol):
    """Protocol for TEE attestation backends."""

    def get_measurement(self) -> bytes:
        """Return the enclave measurement (MRENCLAVE or equivalent)."""
        ...

    def verify_remote(self, expected_measurement: bytes) -> bool:
        """Verify remote attestation against expected measurement."""
        ...


@dataclass(frozen=True)
class AttestationResult:
    is_valid: bool
    measurement: bytes
    provider: str
    error: str = ""


class DstackAttestation:
    """Dstack TEE attestation implementation.

    In production, this calls Dstack's attestation API.
    For hackathon, this is a stub that computes a local
    code hash and compares against expected.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled

    def get_measurement(self) -> bytes:
        """Compute measurement of the sanitizer module.

        Production: read from TEE hardware registers.
        Dev: hash the sanitizer source file as a stand-in.
        """
        if not self._enabled:
            return b"dev-mode-no-attestation"

        # In real deployment, this reads SGX/TDX MRENCLAVE.
        # Stubbed here to hash our own source as demonstration.
        import asp.tee.sanitizer as sanitizer_mod
        import inspect

        source = inspect.getsource(sanitizer_mod)
        return hashlib.sha256(source.encode()).digest()

    def verify_remote(self, expected_measurement: bytes) -> AttestationResult:
        if not self._enabled:
            return AttestationResult(
                is_valid=True,
                measurement=b"dev-mode-no-attestation",
                provider="dstack-stub",
            )

        actual = self.get_measurement()
        is_valid = actual == expected_measurement
        return AttestationResult(
            is_valid=is_valid,
            measurement=actual,
            provider="dstack",
            error="" if is_valid else "measurement mismatch",
        )
