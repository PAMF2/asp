"""
EmbeddingAdapter -- model-agnostic embedding protocol.

Any embedding backend (sentence-transformers, OpenAI, custom)
implements this protocol.  The intent encoder depends only on
the protocol, never on a concrete model.

Dependency Inversion Principle: high-level encoder depends on
abstraction, not on OpenAI SDK or torch.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingAdapter(Protocol):
    """Protocol for text -> vector embedding."""

    @property
    def dim(self) -> int:
        """Dimensionality of the output embedding vectors."""
        ...

    def embed(self, text: str) -> NDArray[np.float64]:
        """Embed a single text string into a dense vector.

        Returns:
            numpy array of shape (dim,), L2-normalized.
        """
        ...

    def embed_batch(self, texts: list[str]) -> NDArray[np.float64]:
        """Embed multiple texts.

        Returns:
            numpy array of shape (len(texts), dim), each row L2-normalized.
        """
        ...


class RandomEmbeddingAdapter:
    """Stub adapter for testing.  Produces deterministic random vectors
    seeded by the hash of the input text."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> NDArray[np.float64]:
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim)
        return vec / np.linalg.norm(vec)  # L2 normalize

    def embed_batch(self, texts: list[str]) -> NDArray[np.float64]:
        return np.stack([self.embed(t) for t in texts])
