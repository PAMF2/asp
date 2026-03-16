"""
ASPDefenseModule -- abstract base class for all defense strategies.

Each concrete module encapsulates domain knowledge about a specific
class of attacks and knows how to mitigate them by rewriting the
sanitized context.

Open/Closed Principle: new modules are added by subclassing.
The DefenseRouter discovers them via registration, not modification.

Single Responsibility: a module does ONE thing -- evaluate a threat
vector and produce a mitigation.  It does not do encoding, validation,
or telemetry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from asp.types import (
    DefenseAction,
    MitigationPayload,
    SanitizedContext,
    ThreatLevel,
    ThreatVector,
)


class ASPDefenseModule(ABC):
    """Abstract base for all defense modules.

    Subclasses must implement:
    - evaluate(): decide if this module can handle the threat
    - mitigate(): produce a MitigationPayload with rewritten context
    - capability_vector: the module's "specialty" in embedding space
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable module name."""
        ...

    @property
    @abstractmethod
    def capability_vector(self) -> NDArray[np.float64]:
        """Vector representing this module's defense specialty.

        The DefenseRouter uses cosine similarity between the threat
        vector and each module's capability vector to select the
        best-matching defense.

        Example: an adversarial roleplay module's capability vector
        would be close to embeddings of roleplay-style attacks.
        """
        ...

    @abstractmethod
    def evaluate(self, threat: ThreatVector) -> float:
        """Score how well this module can handle the given threat.

        Returns:
            Confidence score in [0.0, 1.0].  Higher = better match.
            The router picks the module with the highest score.
        """
        ...

    @abstractmethod
    def mitigate(
        self, threat: ThreatVector, context: SanitizedContext
    ) -> MitigationPayload:
        """Apply this module's defense strategy.

        Must fill in `context.rewritten_prompt` with a safe version
        that the LLM can process without risk.

        Args:
            threat: The encoded threat vector from the intent encoder.
            context: The skeleton SanitizedContext from the sanitizer.

        Returns:
            MitigationPayload containing the defense action and
            the completed SanitizedContext.
        """
        ...

    def handles_threat_level(self, level: ThreatLevel) -> bool:
        """Optional override: restrict module to specific threat levels."""
        return True
