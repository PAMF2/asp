"""
MorphologicalIntentEncoder -- the core of ASP's detection engine.

Projects a raw prompt into latent space and classifies its threat level
based on geometric proximity to known attack signatures.  NO regex,
NO keywords, NO brittle pattern matching.

The "morphological" in the name refers to the shape of intent in
embedding space.  Adversarial prompts that look completely different
on the surface (different words, languages, encodings) still share
geometric structure when their semantic intent is the same.

Cite: adversarial ML literature on latent space clustering of
semantically equivalent attacks.
"""

from __future__ import annotations

from asp.config import ASPConfig
from asp.encoder.attack_signature_db import AttackSignatureDB
from asp.encoder.embedding_adapter import EmbeddingAdapter
from asp.encoder.geometry import find_nearest
from asp.types import ThreatLevel, ThreatVector


class MorphologicalIntentEncoder:
    """Encode a prompt into a ThreatVector via latent space geometry.

    Dependencies (all injected):
    - EmbeddingAdapter: converts text -> vector
    - AttackSignatureDB: known attack embeddings
    - ASPConfig: thresholds

    This class is stateless beyond its injected dependencies.
    """

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        attack_db: AttackSignatureDB,
        config: ASPConfig,
    ) -> None:
        self._adapter = adapter
        self._attack_db = attack_db
        self._config = config

    def encode(self, raw_prompt: str) -> ThreatVector:
        """Project a raw prompt into latent space and classify threat.

        This method accesses the raw prompt text.  After encoding,
        the caller MUST discard the raw prompt.

        Returns:
            ThreatVector with embedding, similarity score, and threat level.
        """
        # Step 1: Embed the prompt
        embedding = self._adapter.embed(raw_prompt)

        # Step 2: Search attack signature DB
        matrix, ids = self._attack_db.get_matrix()

        if matrix is None or len(ids) == 0:
            # Empty attack DB -- no known threats to compare against
            return ThreatVector(
                embedding=embedding,
                max_attack_similarity=0.0,
                nearest_attack_id="",
                threat_level=ThreatLevel.BENIGN,
            )

        nearest_id, max_sim = find_nearest(embedding, matrix, ids)

        # Step 3: Classify threat level based on geometric distance
        threat_level = self._classify(max_sim)

        return ThreatVector(
            embedding=embedding,
            max_attack_similarity=max_sim,
            nearest_attack_id=nearest_id,
            threat_level=threat_level,
        )

    def _classify(self, similarity: float) -> ThreatLevel:
        """Map cosine similarity to threat level using config thresholds.

        The thresholds define concentric zones around attack manifolds:
        - BLOCK/QUARANTINE: inside the manifold (very high similarity)
        - WARN: near the boundary
        - MONITOR: approaching the boundary
        - BENIGN: far from any known attack
        """
        # Thresholds define concentric zones around attack manifolds.
        # With defaults (attack=0.82, monitor=0.65), the zones are:
        #   QUARANTINE: >= 0.92
        #   BLOCK:      >= 0.82
        #   WARN:       >= 0.76  (2/3 of the way from monitor to attack)
        #   MONITOR:    >= 0.65
        #   BENIGN:     < 0.65
        attack = self._config.attack_similarity_threshold
        monitor = self._config.monitor_threshold
        warn = monitor + (attack - monitor) * 2 / 3

        if similarity >= attack + 0.10:
            return ThreatLevel.QUARANTINE
        elif similarity >= attack:
            return ThreatLevel.BLOCK
        elif similarity >= warn:
            return ThreatLevel.WARN
        elif similarity >= monitor:
            return ThreatLevel.MONITOR
        else:
            return ThreatLevel.BENIGN
