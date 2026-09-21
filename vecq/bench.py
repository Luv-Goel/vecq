"""Benchmark harness: build an HNSW, compare it against brute-force search.

``run_benchmark`` returns a dictionary that includes recall, QPS and index
build time at each ``ef_search`` value. ``plot_*`` helpers turn the result
into publication-quality PNG charts using matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .flat import FlatIndex, recall_at_k
from .hnsw import HNSW, HNSWConfig


@dataclass
class BenchmarkRow:
    ef_search: int
    recall_at_10: float
    qps: float
    build_seconds: float


@dataclass
class BenchmarkResult:
    n: int
    dim: int
    metric: str
    flat_qps: float
    flat_build_seconds: float
    hnsw_build_seconds: float
    rows: list[BenchmarkRow]


def make_dataset(
    n: int,
    dim: int,
    *,
    clusters: int = 20,
    seed: int = 0,
    intra_spread: float = 0.6,
    inter_spread: float = 1.2,
) -> np.ndarray:
    """Draw ``n`` unit vectors in ``dim`` dimensions, drawn from ``clusters``."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((clusters, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    centres *= inter_spread

    vectors = np.empty((n, dim), dtype=np.float64)
    for index in range(n):
        centre = centres[rng.integers(clusters)]
        vectors[index] = centre + intra_spread * rng.standard_normal(dim)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.where(norms == 0, 1.0, norms)
    return vectors


def run_benchmark(
    vectors: np.ndarray,
    *,
    k: int = 10,
    ef_search_values: Iterable[int] = (20, 50, 100, 200),
    n_queries: int = 200,
    M: int = 16,
    ef_construction: int = 100,
    metric: str = "cosine",
    seed: int = 0,
) -> BenchmarkResult:
    import time

    flat = FlatIndex.build(vectors, metric=metric)
    rng = np.random.default_rng(seed)
    queries = rng.standard_normal((n_queries, vectors.shape[1]))
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries /= np.where(norms == 0, 1.0, norms)

    t0 = time.perf_counter()
    exact = [flat.search(q, k) for q in queries]
    flat_seconds = time.perf_counter() - t0
    flat_qps = n_queries / flat_seconds if flat_seconds > 0 else float("inf")

    t0 = time.perf_counter()
    hnsw = HNSW(vectors.shape[1], HNSWConfig(M=M, ef_construction=ef_construction, ef_search=ef_search_values[-1], metric=metric, seed=seed))
    hnsw.extend(vectors)
    hnsw_build_seconds = time.perf_counter() - t0

    rows: list[BenchmarkRow] = []
    for ef in ef_search_values:
        hnsw.config.ef_search = ef
        t0 = time.perf_counter()
        approx = [hnsw.search(q, k) for q in queries]
        seconds = time.perf_counter() - t0
        qps = n_queries / seconds if seconds > 0 else float("inf")
        recalls = [recall_at_k(a, e, k) for a, e in zip(approx, exact)]
        rows.append(BenchmarkRow(ef_search=ef, recall_at_10=float(np.mean(recalls)), qps=qps, build_seconds=hnsw_build_seconds))

    return BenchmarkResult(
        n=len(vectors),
        dim=vectors.shape[1],
        metric=metric,
        flat_qps=flat_qps,
        flat_build_seconds=0.0,
        hnsw_build_seconds=hnsw_build_seconds,
        rows=rows,
    )


def plot_recall_vs_qps(result: BenchmarkResult, output: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=120)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    recalls = [row.recall_at_10 for row in result.rows]
    qps = [row.qps for row in result.rows]
    ax.plot(
        recalls,
        qps,
        "o-",
        color="#2563eb",
        markersize=9,
        linewidth=2.2,
        label=f"HNSW (N={result.n}, dim={result.dim})",
    )
    for row in result.rows:
        ax.annotate(
            f"ef={row.ef_search}",
            (row.recall_at_10, row.qps),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color="#1f2937",
        )

    ax.axhline(result.flat_qps, color="#dc2626", linestyle="--", linewidth=1.6, label=f"Flat brute-force ({result.flat_qps:.0f} qps)")
    ax.set_xlabel("Recall@10", fontsize=12)
    ax.set_ylabel("Queries per second", fontsize=12)
    ax.set_title("HNSW vs brute-force: recall vs throughput", fontsize=14, fontweight="bold")
    ax.set_xlim(0.5, 1.02)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def plot_indexing_time(result: BenchmarkResult, output: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    methods = ["Flat (build)", "HNSW (build)"]
    seconds = [result.flat_build_seconds, result.hnsw_build_seconds]
    bars = ax.bar(methods, seconds, color=["#dc2626", "#2563eb"], width=0.5)
    for bar, value in zip(bars, seconds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(seconds) * 0.02, f"{value:.2f}s", ha="center", fontsize=11)
    ax.set_ylabel("Seconds", fontsize=12)
    ax.set_title("Index build time", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def format_table(result: BenchmarkResult) -> str:
    lines = ["ef_search | recall@10 | queries/s"]
    lines.append("-" * 38)
    for row in result.rows:
        lines.append(f"{row.ef_search:>9} | {row.recall_at_10:>10.3f} | {row.qps:>9.0f}")
    return "\n".join(lines)