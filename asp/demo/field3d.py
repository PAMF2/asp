"""
PromptField3D and MemoryRegion — 3-D parametric probability field for
prompt classification and online memory consolidation.

Extracted from viz_server.py so they can be imported without starting
the HTTP demo server or instantiating the full ASP pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import threading


@dataclass
class MemoryRegion:
    """A Gaussian blob in 3D probability space."""
    center: "np.ndarray"   # shape (3,)
    label:  str
    radius: float = 0.20   # sigma of Gaussian kernel
    weight: float = 1.0    # amplified each time a new prompt is absorbed
    count:  int   = 1      # number of prompts absorbed into this region

    def gaussian(self, point: "np.ndarray") -> float:
        """Probability density at `point` under this isotropic Gaussian."""
        d2 = float(np.dot(point - self.center, point - self.center))
        return float(self.weight * np.exp(-d2 / (2.0 * self.radius ** 2)))


class PromptField3D:
    """
    3-D parametric probability field for prompt classification and
    online memory consolidation.

    Public API
    ----------
    n3(prompt, label) -> dict
        Allocate a single prompt in the field.  Returns a result dict with
        keys: position, optimized, energy, distance, nearest_label, action,
              region (MemoryRegion | None)

    field_state() -> list[dict]
        Return all current memory regions (for visualization / audit).
    """

    # -- Axis groupings: which keyword categories feed each abstract dimension --
    _DIM_GROUPS: list[list[str]] = [
        ["social_eng", "roleplay"],       # dim 0: authority / persona
        ["injection",  "smuggling"],      # dim 1: technical bypass
        ["exfiltration", "reframing"],    # dim 2: information extraction
    ]

    # -- Tunable parameters (no hardcoded decision values inside the math) ------
    # tanh scale: converts keyword density -> [0,1).  Higher = sharper gradient.
    _TANH_SCALE: float = 15.0
    # Field energy above this -> prompt is already well-explained -> discard
    NOVELTY_THRESHOLD: float = 0.55
    # Distance below this to nearest center -> too similar -> discard
    MIN_DISTANCE: float = 0.12
    # Gradient-descent step count and learning rate for placement optimisation
    _GD_STEPS: int = 40
    _GD_LR:    float = 0.04

    def __init__(self, adapter: Any, sigma: float = 0.20) -> None:
        self._adapter = adapter
        self._sigma   = sigma
        self._memory: list[MemoryRegion] = []
        self._lock = threading.Lock()
        self._seed_memory()

    # -- Core mathematical components ------------------------------------------

    def _keyword_gradient(self, text: str) -> np.ndarray:
        """
        Parametric 3D word-gradient without hardcoded axis scores.

        For each abstract dimension d:
            hits_d  = sum_{cat in DIM_GROUPS[d]}  |{kw : kw in normalize(text)}|
            total_d = sum_{cat in DIM_GROUPS[d]}  |keywords(cat)|
            score_d = tanh( hits_d / total_d  x  _TANH_SCALE )

        tanh is smooth, bounded [0, 1), and never hard-clips -- the gradient
        compresses naturally near saturation, reflecting diminishing returns
        from additional keyword hits.
        """
        lower = self._adapter._normalize(text)
        scores = np.zeros(3, dtype=np.float64)

        for dim_i, cats in enumerate(self._DIM_GROUPS):
            best_density = 0.0
            for cat in cats:
                kws     = self._adapter._KEYWORDS.get(cat, [])
                hits    = sum(1 for kw in kws if kw.lower() in lower)
                density = hits / max(1, len(kws))
                if density > best_density:
                    best_density = density
            scores[dim_i] = float(np.tanh(best_density * self._TANH_SCALE))

        return scores

    def _field_energy(self, point: np.ndarray) -> float:
        """Sum_i G(point, m_i) -- total Gaussian field energy at a 3D position."""
        return sum(m.gaussian(point) for m in self._memory)

    def _field_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Analytic gradient of the field energy w.r.t. position:
            grad_E(p) = Sum_i  G(p, m_i) * (p - c_i) / sigma_i^2

        Used for gradient-descent placement optimisation.
        """
        grad = np.zeros(3, dtype=np.float64)
        for m in self._memory:
            g    = m.gaussian(point)
            diff = point - m.center
            grad += g * diff / (m.radius ** 2)
        return grad

    def _optimize_placement(self, raw_pos: np.ndarray) -> np.ndarray:
        """
        Gradient descent from raw_pos toward nearest low-energy valley.

        Minimises  E(p) = Sum_i G(p, m_i)  via  p <- p - lr * grad_E(p).
        Clamps to [0, 1]^3 unit cube after each step.
        Terminates early if ||grad_E|| < 1e-4 (flat region reached).
        """
        p  = raw_pos.copy().astype(np.float64)
        lr = self._GD_LR

        for _ in range(self._GD_STEPS):
            g = self._field_gradient(p)
            if np.linalg.norm(g) < 1e-4:
                break
            p = p - lr * g
            p = np.clip(p, 0.0, 1.0)

        return p

    def _nearest_region(self, point: np.ndarray) -> tuple[int, float, str]:
        """Return (index, distance, label) of nearest memory region center."""
        if not self._memory:
            return -1, float("inf"), "none"
        dists = [float(np.linalg.norm(point - m.center)) for m in self._memory]
        idx   = int(np.argmin(dists))
        return idx, dists[idx], self._memory[idx].label

    # -- Public API -------------------------------------------------------------

    def n3(self, prompt: str, label: str = "unknown") -> dict:
        """
        Allocate prompt p1 in the 3D field.

        Steps
        -----
        1. raw_pos   <- keyword_gradient(prompt)         initial 3D position
        2. opt_pos   <- optimize_placement(raw_pos)       energy-valley descent
        3. energy    <- field_energy(opt_pos)             field pressure
        4. nearest   <- nearest_region(opt_pos)           closest existing region
        5. decision  -> NOVEL if energy < threshold AND distance > min_dist
                     -> REDUNDANT otherwise

        Returns dict with full telemetry.  Thread-safe.
        """
        with self._lock:
            raw_pos                    = self._keyword_gradient(prompt)
            opt_pos                    = self._optimize_placement(raw_pos)
            energy                     = self._field_energy(opt_pos)
            near_idx, distance, n_lbl  = self._nearest_region(opt_pos)

            novel = (
                near_idx == -1
                or (distance > self.MIN_DISTANCE and energy < self.NOVELTY_THRESHOLD)
            )

            result: dict = {
                "position":      raw_pos.tolist(),
                "optimized":     opt_pos.tolist(),
                "energy":        round(energy, 6),
                "distance":      round(distance, 6),
                "nearest_label": n_lbl,
                "nearest_idx":   near_idx,
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
                m        = self._memory[near_idx]
                m.center = (m.center * m.count + opt_pos) / (m.count + 1)
                m.count += 1
                m.weight = min(3.0, m.weight + 0.1)
                result["action"] = "discarded"

            return result

    def field_state(self) -> list[dict]:
        """Return all memory regions as serialisable dicts (for visualisation)."""
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

    # -- Seed initial field with canonical prototype regions --------------------

    def _seed_memory(self) -> None:
        """
        Place canonical prototype regions at known positions in the unit cube.
        Seeds give the gradient-descent optimizer initial basins to flow toward.

        Coordinates: (dim0=authority, dim1=bypass, dim2=extraction)
        """
        seeds: list[tuple[str, float, float, float]] = [
            ("benign",       0.03, 0.03, 0.03),
            ("social_eng",   0.90, 0.12, 0.08),
            ("roleplay",     0.78, 0.08, 0.06),
            ("injection",    0.12, 0.92, 0.08),
            ("smuggling",    0.18, 0.82, 0.12),
            ("exfiltration", 0.08, 0.12, 0.92),
            ("reframing",    0.14, 0.18, 0.80),
        ]
        for label, d0, d1, d2 in seeds:
            self._memory.append(MemoryRegion(
                center=np.array([d0, d1, d2], dtype=np.float64),
                label=label,
                radius=self._sigma,
                weight=1.5,
                count=1,
            ))
