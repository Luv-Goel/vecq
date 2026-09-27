"""Brute-force exact nearest-neighbour search.

Single-vector queries compute every distance and sort. Batch queries
vectorise over the index matrix so a 10k×768 search runs as one numpy
matrix multiply instead of 10k Python loops. This module is the
ground-truth recall target that HNSW is benchmarked against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import METRICS, Metric, resolve_metric

Result = tuple[np.ndarray, np.ndarray]  # (distances, ids)


@dataclass
class FlatIndex:
    """A linear-scan index. Search is exact but O(N) per query."""

    vectors: np.ndarray  # shape (n, dim)
    ids: np.ndarray  # shape (n,) — external identifiers, may be ints or strings
    metric: str = "cosine"

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @classmethod
    def build(cls, vectors: np.ndarray, ids: np.ndarray | None = None, metric: str = "cosine") -> "FlatIndex":
        if ids is None:
            ids = np.arange(len(vectors))
        return cls(vectors=np.ascontiguousarray(vectors), ids=np.asarray(ids), metric=metric)

    # -- queries ----------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 10) -> Result:
        """Return ``(k distances, k ids)`` for a single query vector."""
        return _search(self.vectors, self.ids, query, k, METRICS[self.metric])

    def search_batch(self, queries: np.ndarray, k: int = 10) -> list[Result]:
        """Vectorised search for many queries at once."""
        return [_search(self.vectors, self.ids, q, k, METRICS[self.metric]) for q in queries]

    def save(self, path: str) -> None:
        import pickle
        payload = {
            "version": 1,
            "vectors": self.vectors,
            "ids": self.ids,
            "metric": self.metric,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "FlatIndex":
        import pickle
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(vectors=payload["vectors"], ids=payload["ids"], metric=payload["metric"])


def _search(index: np.ndarray, ids: np.ndarray, query: np.ndarray, k: int, metric: Metric) -> Result:
    if len(index) == 0:
        empty = np.empty(0, dtype=ids.dtype), np.empty(0, dtype=np.float64)
        return empty[0], empty[1]
    distances = _vectorised_distances(index, query, metric)
    k_eff = min(k, len(distances))
    # argpartition is O(n); partial sort keeps the top-k at the front.
    top = np.argpartition(distances, k_eff - 1)[:k_eff]
    top = top[np.argsort(distances[top])]
    return distances[top], ids[top]


def _vectorised_distances(index: np.ndarray, query: np.ndarray, metric: Metric) -> np.ndarray:
    if metric is METRICS["cosine"]:
        q_norm = np.linalg.norm(query)
        i_norms = np.linalg.norm(index, axis=1)
        safe_i = np.where(i_norms == 0, 1.0, i_norms)
        sims = (index @ query) / (safe_i * (q_norm if q_norm > 0 else 1.0))
        return 1.0 - sims
    if metric is METRICS["l2"]:
        diffs = index - query
        return np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    if metric is METRICS["ip"]:
        return -(index @ query)
    # Generic fallback for any other metric (slow but correct).
    return np.array([metric(index[i], query) for i in range(len(index))])


def recall_at_k(approx: Result, exact: Result, k: int) -> float:
    """Fraction of the true top-k ids that the approximate result returned."""
    approx_ids = set(approx[1][:k].tolist())
    exact_ids = exact[1][:k].tolist()
    hits = sum(1 for ident in exact_ids if ident in approx_ids)
    return hits / max(1, len(exact_ids))


def queries_per_second(elapsed: float, count: int) -> float:
    """Queries-per-second for a batch run, guarding division by zero."""
    return count / elapsed if elapsed > 0 else float("inf")