"""
PromptFieldND and MemoryRegion — N-dimensional parametric probability field
for prompt classification and online memory consolidation.

Generalized from the original 3D implementation. The number of dimensions,
axis groupings, and seed regions are all configurable. The core math
(Gaussian energy, analytic gradient, gradient-descent placement, novelty
detection) works identically in any ℝⁿ.

Extracted from viz_server.py so it can be imported without starting
the HTTP demo server or instantiating the full ASP pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import threading


# ---------------------------------------------------------------------------
# Default configurations (backwards-compatible 3D)
# ---------------------------------------------------------------------------

DEFAULT_DIM_GROUPS: list[list[str]] = [
    ["social_eng", "roleplay"],       # dim 0: authority / persona
    ["injection",  "smuggling"],      # dim 1: technical bypass
    ["exfiltration", "reframing"],    # dim 2: information extraction
]

DEFAULT_SEEDS: list[tuple[str, list[float]]] = [
    ("benign",       [0.03, 0.03, 0.03]),
    ("social_eng",   [0.90, 0.12, 0.08]),
    ("roleplay",     [0.78, 0.08, 0.06]),
    ("injection",    [0.12, 0.92, 0.08]),
    ("smuggling",    [0.18, 0.82, 0.12]),
    ("exfiltration", [0.08, 0.12, 0.92]),
    ("reframing",    [0.14, 0.18, 0.80]),
]


@dataclass
class MemoryRegion:
    """An isotropic Gaussian blob in N-dimensional probability space.

    G(p) = weight × exp( -‖p - center‖² / (2σ²) )

    Works in any ℝⁿ — the dimension is determined by len(center).
    """
    center: "np.ndarray"   # shape (ndim,)
    label:  str
    radius: float = 0.20   # σ of Gaussian kernel
    weight: float = 1.0    # amplified each time a new prompt is absorbed
    count:  int   = 1      # number of prompts absorbed into this region

    def gaussian(self, point: "np.ndarray") -> float:
        """Probability density at `point` under this isotropic Gaussian."""
        d2 = float(np.dot(point - self.center, point - self.center))
        return float(self.weight * np.exp(-d2 / (2.0 * self.radius ** 2)))


class PromptFieldND:
    """
    N-dimensional parametric probability field for prompt classification
    and online memory consolidation.

    The number of dimensions is determined by len(dim_groups). Each
    dimension maps one or more keyword categories to a scalar score
    via tanh compression. The full math (Gaussian energy, analytic
    gradient, gradient-descent optimization, novelty detection) is
    dimension-agnostic.

    Parameters
    ----------
    adapter : Any
        Embedding adapter with _normalize(text) and _KEYWORDS dict.
    dim_groups : list[list[str]] | None
        Axis groupings — each inner list maps keyword categories to one
        dimension.  Defaults to 3D (authority, bypass, extraction).
    seeds : list[tuple[str, list[float]]] | None
        Seed regions as (label, [coord_0, ..., coord_n]).  Must match
        len(dim_groups).  Defaults to 7 canonical 3D prototypes.
    sigma : float
        Gaussian kernel width for new regions.
    tanh_scale : float
        Steepness of keyword density → score mapping.
    novelty_threshold : float
        Field energy below this → prompt is novel.
    min_distance : float
        Distance above this to nearest center → prompt is novel.
    gd_steps : int
        Gradient descent iterations for placement optimization.
    gd_lr : float
        Gradient descent learning rate.

    Public API
    ----------
    classify(prompt, label) -> dict
        Allocate a single prompt in the field.  Returns result dict with
        keys: position, optimized, energy, distance, nearest_label, action,
              region (MemoryRegion | None), ndim

    field_state() -> list[dict]
        Return all current memory regions (for visualization / audit).
    """

    def __init__(
        self,
        adapter: Any,
        dim_groups: list[list[str]] | None = None,
        seeds: list[tuple[str, list[float]]] | None = None,
        sigma: float = 0.20,
        tanh_scale: float = 15.0,
        novelty_threshold: float = 0.55,
        min_distance: float = 0.12,
        gd_steps: int = 40,
        gd_lr: float = 0.04,
    ) -> None:
        self._adapter = adapter
        self._dim_groups = dim_groups or DEFAULT_DIM_GROUPS
        self._ndim = len(self._dim_groups)
        self._sigma = sigma
        self._tanh_scale = tanh_scale
        self._novelty_threshold = novelty_threshold
        self._min_distance = min_distance
        self._gd_steps = gd_steps
        self._gd_lr = gd_lr
        self._memory: list[MemoryRegion] = []
        self._lock = threading.Lock()
        self._seed_memory(seeds)

    @property
    def ndim(self) -> int:
        """Number of dimensions in this field."""
        return self._ndim

    # -- Core mathematical components ------------------------------------------

    def _keyword_gradient(self, text: str) -> np.ndarray:
        """
        Parametric N-D word-gradient.

        For each dimension d:
            hits_d  = max over cats in dim_groups[d] of (keyword hits / total keywords)
            score_d = tanh( density × tanh_scale )

        tanh is smooth, bounded [0, 1), and compresses near saturation —
        diminishing returns from additional keyword hits.
        """
        lower = self._adapter._normalize(text)
        scores = np.zeros(self._ndim, dtype=np.float64)

        for dim_i, cats in enumerate(self._dim_groups):
            best_density = 0.0
            for cat in cats:
                kws = self._adapter._KEYWORDS.get(cat, [])
                hits = sum(1 for kw in kws if kw.lower() in lower)
                density = hits / max(1, len(kws))
                if density > best_density:
                    best_density = density
            scores[dim_i] = float(np.tanh(best_density * self._tanh_scale))

        return scores

    def _field_energy(self, point: np.ndarray) -> float:
        """E(p) = Σ_i G(p, m_i) — total Gaussian field energy at position p."""
        return sum(m.gaussian(point) for m in self._memory)

    def _field_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Analytic gradient of the field energy w.r.t. position:
            ∇E(p) = Σ_i G(p, m_i) × (p - c_i) / σ_i²

        Used for gradient-descent placement optimization. Works in ℝⁿ.
        """
        grad = np.zeros(self._ndim, dtype=np.float64)
        for m in self._memory:
            g = m.gaussian(point)
            diff = point - m.center
            grad += g * diff / (m.radius ** 2)
        return grad

    def _optimize_placement(self, raw_pos: np.ndarray) -> np.ndarray:
        """
        Gradient descent from raw_pos toward nearest low-energy valley.

        Minimizes E(p) = Σ_i G(p, m_i)  via  p ← p - lr × ∇E(p).
        Clamps to [0, 1]ⁿ unit hypercube after each step.
        Terminates early if ‖∇E‖ < 1e-4 (flat region reached).
        """
        p = raw_pos.copy().astype(np.float64)

        for _ in range(self._gd_steps):
            g = self._field_gradient(p)
            if np.linalg.norm(g) < 1e-4:
                break
            p = p - self._gd_lr * g
            p = np.clip(p, 0.0, 1.0)

        return p

    def _nearest_region(self, point: np.ndarray) -> tuple[int, float, str]:
        """Return (index, distance, label) of nearest memory region center."""
        if not self._memory:
            return -1, float("inf"), "none"
        dists = [float(np.linalg.norm(point - m.center)) for m in self._memory]
        idx = int(np.argmin(dists))
        return idx, dists[idx], self._memory[idx].label

    # -- Public API -----------------------------------------------------------

    def classify(self, prompt: str, label: str = "unknown") -> dict:
        """
        Allocate prompt in the N-D field.

        Steps
        -----
        1. raw_pos  ← keyword_gradient(prompt)        initial ℝⁿ position
        2. opt_pos  ← optimize_placement(raw_pos)     energy-valley descent
        3. energy   ← field_energy(opt_pos)            field pressure
        4. nearest  ← nearest_region(opt_pos)          closest Gaussian blob
        5. decision → NOVEL if energy < threshold AND distance > min_dist
                    → REDUNDANT otherwise (absorbed into nearest region)

        Returns dict with full telemetry. Thread-safe.
        """
        with self._lock:
            raw_pos = self._keyword_gradient(prompt)
            opt_pos = self._optimize_placement(raw_pos)
            energy = self._field_energy(opt_pos)
            near_idx, distance, n_lbl = self._nearest_region(opt_pos)

            novel = (
                near_idx == -1
                or (distance > self._min_distance and energy < self._novelty_threshold)
            )

            result: dict = {
                "position":      raw_pos.tolist(),
                "optimized":     opt_pos.tolist(),
                "energy":        round(energy, 6),
                "distance":      round(distance, 6),
                "nearest_label": n_lbl,
                "nearest_idx":   near_idx,
                "ndim":          self._ndim,
                "action":        None,
                "region":        None,
            }

            if novel:
                m1 = MemoryRegion(
                    center=opt_pos.copy(),
                    label=label,
                    radius=self._sigma,
                    weight=1.0,
                    count=1,
                )
                self._memory.append(m1)
                result["action"] = "memorized"
                result["region"] = m1
            else:
                m = self._memory[near_idx]
                m.center = (m.center * m.count + opt_pos) / (m.count + 1)
                m.count += 1
                m.weight = min(3.0, m.weight + 0.1)
                result["action"] = "discarded"

            return result

    # Backwards-compatible alias
    n3 = classify

    def field_state(self) -> list[dict]:
        """Return all memory regions as serializable dicts."""
        with self._lock:
            return [
                {
                    "center": m.center.tolist(),
                    "label":  m.label,
                    "radius": m.radius,
                    "weight": round(m.weight, 4),
                    "count":  m.count,
                }
                for m in self._memory
            ]

    # -- Seed initial field ---------------------------------------------------

    def _seed_memory(self, seeds: list[tuple[str, list[float]]] | None) -> None:
        """
        Place canonical prototype regions at known positions in [0,1]ⁿ.
        Seeds give the gradient-descent optimizer initial basins to flow toward.
        """
        seed_list = seeds or DEFAULT_SEEDS
        for label, coords in seed_list:
            if len(coords) != self._ndim:
                raise ValueError(
                    f"Seed '{label}' has {len(coords)} coords but field has "
                    f"{self._ndim} dimensions"
                )
            self._memory.append(MemoryRegion(
                center=np.array(coords, dtype=np.float64),
                label=label,
                radius=self._sigma,
                weight=1.5,
                count=1,
            ))


# ---------------------------------------------------------------------------
# Backwards compatibility: PromptField3D is just PromptFieldND with defaults
# ---------------------------------------------------------------------------
PromptField3D = PromptFieldND
