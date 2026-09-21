# Changelog

All notable changes to vecq are recorded here. Versions follow [semantic versioning](https://semver.org/).

## 1.0.0 — 2026-09-21

First public release. Implements:

- **Flat brute-force index** (`FlatIndex`) with a vectorised search over the whole matrix and a vectorised cosine / L2 / inner-product fast path. Used as the recall ground truth.
- **HNSW index** (`HNSW`, `HNSWConfig`) with the classic greedy descent + beam-search layers, neighbour-diversity selection heuristic, per-layer link caps and batched distance computation in the hot loop. Search supports a metadata-filter predicate.
- **Distance metrics** for cosine, L2 and inner-product; every metric returns a *distance* so consumers can sort and threshold uniformly.
- **Persistence** via a versioned pickle payload covering the vectors, per-node neighbour lists, entry point and max layer.
- **Benchmark harness** that draws a clustered synthetic dataset, runs the index at several `ef_search` values, and produces two matplotlib charts (recall vs QPS, build time).
- **CLI** with `search`, `bench`, `demo` and `info` subcommands.
- **19 pytest cases** covering metrics, exact search, HNSW correctness against brute-force, persistence round-trip, metadata filtering, and the benchmark harness.