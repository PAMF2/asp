# ASP — Alignment Security Protocol

A defense-in-depth architecture for LLM safety that moves enforcement from surface-form pattern matching to **geometric intent detection** backed by hardware isolation and cryptographic consensus.

> Hackathon submission: IC3/Encode Club Shape Rotator '25

## The Problem

Existing LLM safety defenses (keyword filters, regex blocklists, fine-tuned refusal classifiers) all operate on **surface form**. Attacks that paraphrase, encode (Base64, ROT13, hex), translate, or compositionally obfuscate their intent bypass them trivially.

ASP replaces surface-form matching with five defense layers:

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────┐
│  1. TEE Boundary                            │
│     Raw prompt destroyed after encoding     │
├─────────────────────────────────────────────┤
│  2. Morphological Intent Encoder            │
│     Cosine similarity in latent space       │
│     → ThreatVector (BENIGN..QUARANTINE)     │
├─────────────────────────────────────────────┤
│  3. Defense Router                          │
│     Best-match module via capability vecs   │
│     → MitigationPayload                     │
├─────────────────────────────────────────────┤
│  4. Threshold Validator (N-of-M)            │
│     Shamir shares + quorum consensus        │
│     → VERIFIED_IMMUNITY or REJECTED         │
├─────────────────────────────────────────────┤
│  5. Gossip Protocol                         │
│     Epidemic vaccine propagation            │
│     O(log N) convergence                    │
└─────────────────────────────────────────────┘
    │
    ▼
  LLM receives only SanitizedContext (never raw prompt)
```

## Key Invariants

- **Raw prompt non-disclosure**: the raw prompt is a local variable inside the TEE boundary, explicitly deleted after encoding. The LLM only ever sees `SanitizedContext`.
- **Surface-form invariance**: classification happens in embedding space — attacks with the same semantic intent cluster together regardless of paraphrase, encoding, or language.
- **Byzantine fault tolerance**: a single compromised validator cannot forge a `VERIFIED_IMMUNITY` verdict (N-of-M threshold signatures required).
- **Idempotent vaccine propagation**: duplicate attack signatures are deduplicated by `signature_hash`; gossip converges in O(log N) rounds.

## Modules

| Module | Purpose |
|--------|---------|
| `encoder/` | Morphological intent encoder — projects prompts to latent space and classifies threat via geometry |
| `defense/` | Multi-strategy defense router — selects mitigation module by cosine similarity to capability vectors |
| `tee/` | Trusted Execution Environment boundary — sanitizer, pipeline orchestrator, remote attestation |
| `threshold/` | N-of-M consensus — Shamir secret sharing, validator nodes, quorum aggregation |
| `gossip/` | Epidemic gossip protocol — vaccine propagation, peer management, pluggable transport |
| `llm/` | Model-agnostic LLM interface — receives only `SanitizedContext` (OpenAI, Llama adapters) |
| `telemetry/` | JSON-RPC 2.0 telemetry — structured event emission for threat, mitigation, and threshold events |
| `demo/` | Live visualization server — gossip network, latent space scatter, TEE x-ray |

## Threat Levels & Defense Actions

```
ThreatLevel:    BENIGN → MONITOR → WARN → BLOCK → QUARANTINE
DefenseAction:  PASS_THROUGH | CONTEXT_AUGMENT | ROLEPLAY_REDIRECT | SANITIZE_AND_REWRITE | FULL_BLOCK
Verdict:        UNTRUSTED → PENDING_VALIDATION → VERIFIED_IMMUNITY | REJECTED
```

## Quick Start

```bash
# Install (core has no ML framework dependencies — only numpy)
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run the 100-prompt attack suite
python test100.py

# Launch the demo visualization server (port 7475)
python -m asp.demo.viz_server
```

## Configuration

All parameters are sourced from environment variables with sensible defaults (see `config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ASP_EMBEDDING_DIM` | 768 | Embedding dimensionality |
| `ASP_ATTACK_THRESHOLD` | 0.82 | Cosine similarity threshold for BLOCK |
| `ASP_MONITOR_THRESHOLD` | 0.65 | Cosine similarity threshold for MONITOR |
| `ASP_THRESHOLD_N` | 2 | Minimum signatures for quorum |
| `ASP_THRESHOLD_M` | 3 | Total validator nodes |
| `ASP_GOSSIP_FANOUT` | 3 | Peers per gossip round |
| `ASP_GOSSIP_INTERVAL` | 1.0 | Seconds between gossip rounds |

## Testing

```bash
# Unit + integration tests
pytest -v

# 100-prompt attack suite (8 categories: benign, roleplay, injection,
# exfiltration, smuggling, social engineering, reframing, borderline)
python test100.py
```

## Architecture Details

See `asp-whitepaper.tex` for formal treatment including:
- Proof that a single compromised validator cannot produce `VERIFIED_IMMUNITY`
- Surface-form invariance theorem for the morphological encoder
- Gossip convergence analysis
- Alignment tax trilemma (precision vs. recall vs. latency)

## Dependencies

- **Core**: `numpy >= 1.26` (no ML frameworks in the critical path)
- **Dev**: `pytest >= 8.0`, `pytest-asyncio >= 0.23`
- **Optional (LLM)**: `openai >= 1.0`, `aiohttp >= 3.9`

## License

Hackathon submission — IC3/Encode Club Shape Rotator '25.
