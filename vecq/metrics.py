"""Distance metrics for vector search.

Each metric returns a *distance* — smaller is closer — so the same code
can sort and threshold across metrics without special cases. For the
``"ip"`` (inner product) metric, distance is the negative of the raw
dot product, which makes "highest similarity" equivalent to "lowest
distance".
"""

from __future__ import annotations

from typing import Callable

import numpy as np


Metric = Callable[[np.ndarray, np.ndarray], float]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cos(a, b). Range: [0, 2]; 0 means identical direction."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance. Range: [0, inf)."""
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def inner_product_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Negative of the dot product. Smaller means more similar."""
    return -float(np.dot(a, b))


METRICS: dict[str, Metric] = {
    "cosine": cosine_distance,
    "l2": l2_distance,
    "ip": inner_product_distance,
}


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise every row. Returns a fresh array; never mutates input."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return vectors / safe


def resolve_metric(name: str) -> Metric:
    if name not in METRICS:
        raise ValueError(f"unknown metric {name!r}; choose from {sorted(METRICS)}")
    return METRICS[name]