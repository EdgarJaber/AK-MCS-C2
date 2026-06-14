"""Reliability benchmark problems.

A :class:`ReliabilityProblem` bundles a limit-state function ``g`` (failure
iff ``g(x) <= 0``), a sampler from the input distribution, the input
dimension and an optional reference failure probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

__all__ = [
    "ReliabilityProblem",
    "four_branch",
    "modified_rastrigin",
    "linear_oscillator",
    "get_problem",
    "PROBLEMS",
]


@dataclass
class ReliabilityProblem:
    name: str
    dim: int
    g: Callable[[np.ndarray], np.ndarray]
    sampler: Callable[[int, np.random.Generator], np.ndarray]
    pf_ref: Optional[float] = None
    description: str = ""

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return self.sampler(n, rng)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.g(np.atleast_2d(X)), dtype=float).reshape(-1)


# ---------------------------------------------------------------------------
# 2D four-branch series system
# ---------------------------------------------------------------------------

def four_branch(k: float = 6.0) -> ReliabilityProblem:
    """Two-dimensional four-branch series system (Waarts, 2000)."""

    def g(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        x1, x2 = X[:, 0], X[:, 1]
        g1 = 3.0 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / np.sqrt(2.0)
        g2 = 3.0 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / np.sqrt(2.0)
        g3 = (x1 - x2) + k / np.sqrt(2.0)
        g4 = (x2 - x1) + k / np.sqrt(2.0)
        return np.minimum.reduce([g1, g2, g3, g4])

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, 2))

    pf_ref = {6.0: 4.46e-3, 7.0: 2.23e-3}.get(float(k))
    return ReliabilityProblem(
        name=f"branch2d_k{int(k)}",
        dim=2,
        g=g,
        sampler=sampler,
        pf_ref=pf_ref,
        description=f"2D four-branch series system, k={k}",
    )


# ---------------------------------------------------------------------------
# 2D modified Rastrigin
# ---------------------------------------------------------------------------

def modified_rastrigin() -> ReliabilityProblem:
    """Modified Rastrigin function: disconnected, multimodal failure domain."""

    def g(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return 10.0 - np.sum(X**2 - 5.0 * np.cos(2.0 * np.pi * X), axis=1)

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, 2))

    return ReliabilityProblem(
        name="rastrigin2d",
        dim=2,
        g=g,
        sampler=sampler,
        pf_ref=7.3e-2,
        description="2D modified Rastrigin, disjoint failure islands",
    )


# ---------------------------------------------------------------------------
# 6D oscillator
# ---------------------------------------------------------------------------

def linear_oscillator() -> ReliabilityProblem:
    """Single-DOF oscillator under transient excitation (Echard et al., 2011)."""

    def g(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        c1, c2, m, r, t1, F1 = (X[:, j] for j in range(6))
        w0sq = (c1 + c2) / m
        return 3.0 * r - np.abs(2.0 * F1 / (m * w0sq) * np.sin(w0sq * t1 / 2.0))

    means = np.array([1.0, 0.1, 1.0, 0.5, 1.0, 1.0])
    stds = np.array([0.1, 0.01, 0.05, 0.05, 0.2, 0.2])
    # ordering: c1, c2, m, r, t1, F1

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return means[None, :] + stds[None, :] * rng.standard_normal((n, 6))

    return ReliabilityProblem(
        name="oscillator6d",
        dim=6,
        g=g,
        sampler=sampler,
        pf_ref=3.9e-2,
        description="6D nonlinear oscillator",
    )


PROBLEMS: Dict[str, Callable[[], ReliabilityProblem]] = {
    "branch2d_k6": lambda: four_branch(6.0),
    "branch2d_k7": lambda: four_branch(7.0),
    "rastrigin2d": modified_rastrigin,
    "oscillator6d": linear_oscillator,
}


def get_problem(name: str) -> ReliabilityProblem:
    try:
        return PROBLEMS[name]()
    except KeyError as e:
        raise KeyError(
            f"Unknown problem '{name}'. Available: {sorted(PROBLEMS)}"
        ) from e
