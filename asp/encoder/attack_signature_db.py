"""
AttackSignatureDB -- vector store of known attack embeddings.

This is the "immune system memory" of the ASP network.  Each entry
is the embedding of a known jailbreak/attack prompt, keyed by a
content hash.

The DB grows over time as vaccines propagate via gossip.  It is
append-only and idempotent (adding the same signature twice is a no-op).

Storage: in-memory numpy matrix for hackathon.  Production would
back this with FAISS or pgvector.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asp.encoder.geometry import cosine_similarity_matrix


@dataclass
class AttackSignature:
    """A single known attack pattern in the DB."""
    signature_id: str          # SHA-256 of the embedding
    embedding: NDArray[np.float64]
    category: str = "unknown"  # e.g., "roleplay", "context_injection"
    source: str = ""           # which node discovered it


class AttackSignatureDB:
    """Thread-safe vector store for attack signatures.

    Supports:
    - O(1) dedup by signature_id
    - O(n) nearest-neighbor search (fine for hackathon scale)
    - Append from gossip vaccines

    For production, replace the numpy matrix with a proper ANN index.
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim
        self._lock = threading.Lock()
        self._signatures: dict[str, AttackSignature] = {}
        self._matrix: NDArray[np.float64] | None = None
        self._ids: list[str] = []
        self._dirty = True  # matrix needs rebuild

    @property
    def size(self) -> int:
        return len(self._signatures)

    def add(self, signature: AttackSignature) -> bool:
        """Add a signature.  Returns True if new, False if duplicate."""
        with self._lock:
            if signature.signature_id in self._signatures:
                return False
            self._signatures[signature.signature_id] = signature
            self._dirty = True
            return True

    def add_batch(self, signatures: list[AttackSignature]) -> int:
        """Add multiple signatures.  Returns count of new additions."""
        added = 0
        with self._lock:
            for sig in signatures:
                if sig.signature_id not in self._signatures:
                    self._signatures[sig.signature_id] = sig
                    added += 1
            if added > 0:
                self._dirty = True
        return added

    def search(
        self, query: NDArray[np.float64], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Find the top_k most similar attack signatures.

        Returns:
            List of (signature_id, cosine_similarity) tuples, descending.
        """
        with self._lock:
            self._rebuild_if_dirty()

            if self._matrix is None or self._matrix.shape[0] == 0:
                return []

            sims = cosine_similarity_matrix(query, self._matrix)
            k = min(top_k, len(sims))
            top_indices = np.argsort(sims)[-k:][::-1]

            return [(self._ids[i], float(sims[i])) for i in top_indices]

    def contains(self, signature_id: str) -> bool:
        with self._lock:
            return signature_id in self._signatures

    def get_matrix(self) -> tuple[NDArray[np.float64] | None, list[str]]:
        """Return the current matrix and ID list.  For encoder use."""
        with self._lock:
            self._rebuild_if_dirty()
            return self._matrix, list(self._ids)

    def _rebuild_if_dirty(self) -> None:
        """Rebuild the numpy matrix from the signatures dict."""
        if not self._dirty:
            return

        if len(self._signatures) == 0:
            self._matrix = None
            self._ids = []
        else:
            self._ids = list(self._signatures.keys())
            self._matrix = np.stack(
                [self._signatures[sid].embedding for sid in self._ids]
            )

        self._dirty = False
