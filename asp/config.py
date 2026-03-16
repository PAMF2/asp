"""
ASP Runtime Configuration

Immutable after construction. All values sourced from environment
variables with sensible defaults for development/hackathon use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ASPConfig:
    """Single source of truth for all tunable parameters."""

    # --- Encoder ---
    embedding_dim: int = 768
    attack_similarity_threshold: float = 0.82  # cosine sim above this = threat
    monitor_threshold: float = 0.65            # above this = MONITOR level

    # --- Threshold validation ---
    threshold_n: int = 2         # N signatures required
    threshold_m: int = 3         # M total validator nodes
    validation_timeout_s: float = 5.0

    # --- Gossip ---
    gossip_fanout: int = 3       # peers per round
    gossip_interval_s: float = 1.0
    gossip_max_rounds: int = 20  # convergence bound

    # --- TEE ---
    tee_attestation_enabled: bool = False  # disabled for local dev
    tee_provider: str = "dstack"

    # --- LLM ---
    llm_provider: str = "openai"  # openai | llama | custom
    llm_model: str = "gpt-4"
    llm_api_key: str = ""

    # --- Telemetry ---
    telemetry_endpoint: str = "http://localhost:8545"
    telemetry_batch_size: int = 10

    @classmethod
    def from_env(cls) -> ASPConfig:
        """Load config from environment variables, falling back to defaults."""
        return cls(
            embedding_dim=int(os.getenv("ASP_EMBEDDING_DIM", "768")),
            attack_similarity_threshold=float(
                os.getenv("ASP_ATTACK_THRESHOLD", "0.82")
            ),
            monitor_threshold=float(os.getenv("ASP_MONITOR_THRESHOLD", "0.65")),
            threshold_n=int(os.getenv("ASP_THRESHOLD_N", "2")),
            threshold_m=int(os.getenv("ASP_THRESHOLD_M", "3")),
            validation_timeout_s=float(os.getenv("ASP_VALIDATION_TIMEOUT", "5.0")),
            gossip_fanout=int(os.getenv("ASP_GOSSIP_FANOUT", "3")),
            gossip_interval_s=float(os.getenv("ASP_GOSSIP_INTERVAL", "1.0")),
            gossip_max_rounds=int(os.getenv("ASP_GOSSIP_MAX_ROUNDS", "20")),
            tee_attestation_enabled=os.getenv("ASP_TEE_ATTESTATION", "false").lower()
            == "true",
            tee_provider=os.getenv("ASP_TEE_PROVIDER", "dstack"),
            llm_provider=os.getenv("ASP_LLM_PROVIDER", "openai"),
            llm_model=os.getenv("ASP_LLM_MODEL", "gpt-4"),
            llm_api_key=os.getenv("ASP_LLM_API_KEY", ""),
            telemetry_endpoint=os.getenv(
                "ASP_TELEMETRY_ENDPOINT", "http://localhost:8545"
            ),
            telemetry_batch_size=int(os.getenv("ASP_TELEMETRY_BATCH_SIZE", "10")),
        )
