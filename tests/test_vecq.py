"""Tests for vecq — metrics, flat, HNSW, persistence, metadata, and the benchmark harness."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vecq import HNSW, HNSWConfig, FlatIndex
from vecq.bench import format_table, make_dataset, run_benchmark
from vecq.metrics import cosine_distance, inner_product_distance, l2_distance, normalize
from vecq.flat import recall_at_k


def _unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    vector = rng.standard_normal(dim)
    vector /= np.linalg.norm(vector)
    return vector


class TestMetrics:
    def test_cosine_zero_for_identical_unit_vectors(self):
        rng = np.random.default_rng(0)
        vector = _unit(rng, 32)
        assert cosine_distance(vector, vector) == pytest.approx(0.0, abs=1e-12)

    def test_cosine_one_for_zero_vector(self):
        zero = np.zeros(8)
        v = _unit(np.random.default_rng(0), 8)
        assert cosine_distance(zero, v) == 1.0

    def test_l2_matches_euclidean_norm(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 3.0])
        assert l2_distance(a, b) == pytest.approx(5.0)

    def test_inner_product_is_negative_dot(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert inner_product_distance(a, b) == pytest.approx(-11.0)

    def test_normalize_handles_zero_vector(self):
        vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        out = normalize(vectors)
        assert out[0] == pytest.approx([0.6, 0.8, 0.0])
        assert out[1] == pytest.approx([0.0, 0.0, 0.0])

    def test_normalize_does_not_mutate_input(self):
        original = np.array([[3.0, 4.0]])
        snapshot = original.copy()
        normalize(original)
        assert np.array_equal(original, snapshot)


class TestFlat:
    def test_search_on_empty_index(self):
        index = FlatIndex.build(np.empty((0, 4)))
        distances, ids = index.search(np.zeros(4), k=10)
        assert len(distances) == 0 and len(ids) == 0

    def test_search_returns_top_k_by_distance(self):
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((100, 16))
        index = FlatIndex.build(vectors, metric="l2")
        # Use a real (non-zero) query so distances are well-defined and unique-ish.
        query = rng.standard_normal(16)
        distances, ids = index.search(query, k=5)
        assert distances[0] <= distances[1] <= distances[2] <= distances[3] <= distances[4]
        expected_nearest = int(np.argmin(np.linalg.norm(vectors - query, axis=1)))
        assert int(ids[0]) == expected_nearest

    def test_recall_at_k_full_overlap(self):
        a = (np.array([0.1, 0.2, 0.3]), np.array([10, 20, 30]))
        b = (np.array([0.5, 0.6, 0.7]), np.array([10, 20, 99]))
        assert recall_at_k(a, b, 3) == pytest.approx(2 / 3)

    def test_save_and_load_round_trip(self, tmp_path):
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
        ids = np.array([10, 20])
        index = FlatIndex.build(vectors, ids, metric="l2")
        path = tmp_path / "flat_index.pkl"
        index.save(str(path))
        
        restored = FlatIndex.load(str(path))
        assert len(restored) == 2
        assert restored.dim == 2
        assert restored.metric == "l2"
        np.testing.assert_array_equal(restored.vectors, vectors)
        np.testing.assert_array_equal(restored.ids, ids)

class TestHNSW:
    def _build_small(self, n: int = 200, dim: int = 16, seed: int = 0) -> tuple[HNSW, np.ndarray]:
        rng = np.random.default_rng(seed)
        vectors = rng.standard_normal((n, dim)).astype(np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index = HNSW(dim, HNSWConfig(M=8, ef_construction=40, ef_search=20, metric="cosine", seed=seed))
        index.extend(vectors)
        return index, vectors

    def test_search_on_empty_index_returns_empty(self):
        index = HNSW(dim=4)
        distances, ids = index.search(np.zeros(4), k=10)
        assert len(distances) == 0 and len(ids) == 0

    def test_recall_is_high_against_brute_force(self):
        index, vectors = self._build_small(n=300, dim=32)
        flat = FlatIndex.build(vectors, metric="cosine")
        # A real (non-zero) query so cosine distances are all distinct.
        query = np.random.default_rng(42).standard_normal(32)
        approx = index.search(query, k=10)
        exact = flat.search(query, k=10)
        assert recall_at_k(approx, exact, 10) >= 0.8

    def test_external_ids_round_trip(self):
        index, _ = self._build_small(n=50, dim=8)
        rng = np.random.default_rng(7)
        custom_ids = [f"doc-{i}" for i in range(50)]
        # rebuild with custom ids by adding again
        index2 = HNSW(8, HNSWConfig(M=8, ef_construction=20, ef_search=10, metric="cosine"))
        rng = np.random.default_rng(1)
        vectors = rng.standard_normal((50, 8))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for vector, ident in zip(vectors, custom_ids):
            index2.add(vector, ident)
        _, ids = index2.search(vectors[0], k=3)
        for ident in ids:
            assert ident in custom_ids

    def test_save_and_load_round_trip(self, tmp_path):
        index, vectors = self._build_small(n=150, dim=12)
        path = tmp_path / "index.pkl"
        index.save(str(path))
        restored = HNSW.load(str(path))
        assert len(restored) == len(index)
        query = np.zeros(12)
        a, _ = index.search(query, k=5)
        b, _ = restored.search(query, k=5)
        np.testing.assert_allclose(a, b)

    def test_filter_drops_non_matching_ids(self):
        rng = np.random.default_rng(2)
        vectors = rng.standard_normal((200, 8)).astype(np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index = HNSW(8, HNSWConfig(M=8, ef_construction=20, ef_search=20, metric="cosine"))
        for offset, vector in enumerate(vectors):
            index.add(vector, external_id=offset)
        only_even = lambda ident: int(ident) % 2 == 0
        distances, ids = index.search(np.zeros(8), k=5, filter_fn=only_even)
        assert all(int(i) % 2 == 0 for i in ids)
        assert len(ids) == 5

    def test_stats_reports_layer_distribution(self):
        index, _ = self._build_small(n=200, dim=8)
        stats = index.stats()
        assert stats["nodes"] == 200
        assert stats["layers"] >= 1
        # Layer 0 must contain every node; upper layers must have fewer.
        assert stats["nodes_per_layer"].get(0, 0) == 200
        upper_layers = {k: v for k, v in stats["nodes_per_layer"].items() if k > 0}
        for count in upper_layers.values():
            assert count <= 200


class TestBenchmark:
    def test_make_dataset_shape_and_unit_norm(self):
        vectors = make_dataset(n=100, dim=16, seed=42)
        assert vectors.shape == (100, 16)
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(100), atol=1e-10)

    def test_run_benchmark_returns_results_with_expected_shape(self):
        vectors = make_dataset(n=300, dim=16, seed=0)
        result = run_benchmark(vectors, ef_search_values=(20, 50), n_queries=20, M=8, ef_construction=40)
        assert result.n == 300 and result.dim == 16
        assert len(result.rows) == 2
        # Higher ef_search must not reduce recall on this easy dataset.
        assert result.rows[1].recall_at_10 >= result.rows[0].recall_at_10
        for row in result.rows:
            assert 0.0 <= row.recall_at_10 <= 1.0
            assert row.qps > 0

    def test_format_table_is_readable(self):
        vectors = make_dataset(n=100, dim=8, seed=0)
        result = run_benchmark(vectors, ef_search_values=(20,), n_queries=10, M=8, ef_construction=20)
        text = format_table(result)
        assert "recall@10" in text
        assert "queries/s" in text
        assert "20" in text

    def test_plot_writes_a_png(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        vectors = make_dataset(n=200, dim=12, seed=0)
        result = run_benchmark(vectors, ef_search_values=(20, 50), n_queries=20, M=8, ef_construction=20)
        from vecq.bench import plot_recall_vs_qps

        output = plot_recall_vs_qps(result, str(tmp_path / "chart.png"))
        assert Path(output).exists() and Path(output).stat().st_size > 0