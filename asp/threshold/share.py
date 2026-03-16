"""
SecretShare -- Shamir-style secret sharing for threshold signatures.

Implements polynomial interpolation over a finite field to split
a secret into M shares where any N shares can reconstruct it.

Ref: Shamir, A. "How to Share a Secret" (1979).
Ref: Thetacrypt -- threshold cryptography for distributed consensus.

Uses a large prime field for arithmetic.  numpy is used only for
convenience in polynomial evaluation; the core math is integer
arithmetic over Z_p.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass


# Large prime for finite field arithmetic (256-bit)
# In production, use a standardized curve's order.
FIELD_PRIME = (1 << 256) - 189


@dataclass(frozen=True)
class Share:
    """A single Shamir share: (index, value) pair."""
    index: int      # x-coordinate (1-indexed, never 0)
    value: int      # y-coordinate (polynomial evaluated at index)


def split_secret(secret: bytes, n: int, m: int) -> list[Share]:
    """Split a secret into M shares, any N of which reconstruct it.

    Args:
        secret: The secret to split (arbitrary bytes).
        n: Threshold -- minimum shares needed for reconstruction.
        m: Total number of shares to generate.

    Returns:
        List of M Share objects.

    Raises:
        ValueError: if n > m or n < 1.
    """
    if n < 1 or n > m:
        raise ValueError(f"Invalid threshold: n={n}, m={m}")

    # Convert secret to integer
    secret_int = int.from_bytes(secret, "big") % FIELD_PRIME

    # Generate random polynomial coefficients: a_0 = secret, a_1..a_{n-1} random
    coefficients = [secret_int] + [
        secrets.randbelow(FIELD_PRIME) for _ in range(n - 1)
    ]

    # Evaluate polynomial at x = 1, 2, ..., m
    shares = []
    for x in range(1, m + 1):
        y = _eval_polynomial(coefficients, x)
        shares.append(Share(index=x, value=y))

    return shares


def reconstruct_secret(shares: list[Share], n: int) -> bytes:
    """Reconstruct the secret from N or more shares using Lagrange interpolation.

    Args:
        shares: At least N shares.
        n: The threshold (for validation).

    Returns:
        The original secret bytes.

    Raises:
        ValueError: if fewer than n shares provided.
    """
    if len(shares) < n:
        raise ValueError(f"Need at least {n} shares, got {len(shares)}")

    # Use exactly n shares — sort by index for deterministic reconstruction
    subset = sorted(shares, key=lambda s: s.index)[:n]

    # Lagrange interpolation at x=0 to recover a_0 (the secret)
    secret_int = 0
    for i, share_i in enumerate(subset):
        numerator = 1
        denominator = 1
        for j, share_j in enumerate(subset):
            if i == j:
                continue
            numerator = (numerator * (-share_j.index)) % FIELD_PRIME
            denominator = (denominator * (share_i.index - share_j.index)) % FIELD_PRIME

        lagrange = (numerator * _mod_inverse(denominator, FIELD_PRIME)) % FIELD_PRIME
        secret_int = (secret_int + share_i.value * lagrange) % FIELD_PRIME

    # Convert back to bytes (32 bytes = 256 bits)
    return secret_int.to_bytes(32, "big")


def sign_verdict(node_secret: bytes, request_id: str, verdict_str: str) -> bytes:
    """Produce a deterministic signature over a verdict using HMAC-SHA256.

    In production, this would be a proper threshold signature scheme
    (e.g., threshold BLS or threshold ECDSA).  For hackathon, we use
    HMAC as a stand-in that demonstrates the protocol mechanics.
    """
    import hmac

    message = f"{request_id}:{verdict_str}".encode()
    return hmac.new(node_secret, message, hashlib.sha256).digest()


def _eval_polynomial(coefficients: list[int], x: int) -> int:
    """Evaluate polynomial at x using Horner's method over Z_p."""
    result = 0
    for coeff in reversed(coefficients):
        result = (result * x + coeff) % FIELD_PRIME
    return result


def _mod_inverse(a: int, p: int) -> int:
    """Modular multiplicative inverse using Fermat's little theorem.
    Requires p to be prime."""
    return pow(a, p - 2, p)
