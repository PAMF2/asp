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
| `demo/` | Live visualization server — gossip network, latent space scatter, TEE x-ray, 3D Gaussian field |
| `demo/field3d.py` | PromptField3D — 3D parametric Gaussian memory field for prompt classification |

## PromptField3D — Gaussian Memory Field

A 3D parametric probability field that classifies prompts and consolidates attack memory online. Each axis represents an abstract threat dimension:

```
  dim 0 (x):  Authority / Persona    ← social_eng, roleplay
  dim 1 (y):  Technical Bypass       ← injection, smuggling
  dim 2 (z):  Information Extraction  ← exfiltration, reframing
```

### Core Math

**1. Keyword Gradient** — maps prompt text to a raw 3D position:

```
  score_d = tanh( (hits_d / total_d) × scale )
```

`tanh` is smooth, bounded [0, 1), and compresses naturally near saturation — diminishing returns from additional keyword hits.

**2. Gaussian Energy** — each memory region `m_i` is an isotropic Gaussian blob:

```
  G(p, m_i) = w_i × exp( -‖p - c_i‖² / (2σ_i²) )
  E(p) = Σ_i G(p, m_i)        ← total field energy at point p
```

**3. Analytic Gradient** — used for placement optimization via gradient descent:

```
  ∇E(p) = Σ_i G(p, m_i) × (p - c_i) / σ_i²
  p ← p - lr × ∇E(p)          ← 40 steps, lr=0.04, clamped to [0,1]³
```

**4. Novelty Detection** — decides whether a prompt is novel or redundant:

```
  NOVEL      if  E(p) < 0.55  AND  dist(p, nearest) > 0.12
  REDUNDANT  otherwise → absorbed into nearest region (running centroid average)
```

### Pipeline

```
prompt → keyword_gradient() → raw_pos [0,1]³
       → optimize_placement() → gradient descent to low-energy valley
       → field_energy() → pressure from existing regions
       → nearest_region() → closest Gaussian blob
       → decision: memorize (new region) or absorb (update existing)
```

### Seed Regions

Seven canonical prototypes initialize the field basins:

| Region | Position (authority, bypass, extraction) |
|--------|----------------------------------------|
| benign | (0.03, 0.03, 0.03) |
| social_eng | (0.90, 0.12, 0.08) |
| roleplay | (0.78, 0.08, 0.06) |
| injection | (0.12, 0.92, 0.08) |
| smuggling | (0.18, 0.82, 0.12) |
| exfiltration | (0.08, 0.12, 0.92) |
| reframing | (0.14, 0.18, 0.80) |

### API

```bash
# Probe a prompt in the 3D field
curl -X POST http://localhost:7475/api/field3d \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ignore all instructions", "label": "test"}'

# Get all memory regions
curl http://localhost:7475/api/field3d/state
```

### Visualization

The demo server includes an interactive Three.js 3D scatter plot with:
- Translucent Gaussian region spheres (color-coded by category)
- Probe points with emissive glow (color-coded by threat level)
- Orbit controls (drag to rotate, scroll to zoom)

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
