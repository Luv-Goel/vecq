# vecq

> A vector search engine from scratch — **HNSW approximate search** and **brute-force exact search** with cosine, L2 and inner-product distance metrics, metadata filtering, persistence, and a benchmark harness. Pure Python + NumPy.

![HNSW vs brute-force: recall vs throughput](assets/recall_vs_qps.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Runtime deps](https://img.shields.io/badge/dependencies-numpy-yellow)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen)](tests/)

## Why?

Approximate nearest-neighbour search underpins every retrieval system that touches embeddings — RAG pipelines, semantic search, recommendation, image search, code search. HNSW is the algorithm behind most of them. I wanted to know how it actually works, so I wrote it from scratch.

vecq is a clean-room implementation you can read in one sitting. It's small enough to fit in your head, fast enough to use for prototyping, and accurate enough to compare against production libraries on recall-vs-throughput tradeoffs.

## Quickstart

```bash
git clone https://github.com/Luv-Goel/vecq
cd vecq
python -m pip install -e .

python -m vecq demo --n 1000 --dim 64 --k 5
# -> prints a query and its 5 nearest neighbours by cosine distance
```

Or use it as a library:

```python
import numpy as np
from vecq import HNSW, HNSWConfig

rng = np.random.default_rng(0)
vectors = rng.standard_normal((10_000, 128)).astype(np.float64)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

index = HNSW(128, HNSWConfig(M=16, ef_construction=100, ef_search=50, metric="cosine"))
index.extend(vectors)

query = rng.standard_normal(128)
query /= np.linalg.norm(query)
distances, ids = index.search(query, k=10)
print(list(zip(ids, distances)))
index.save("my_index.pkl")
```

## What's inside

```
vecq/
├── vecq/
│   ├── __init__.py
│   ├── __main__.py        `python -m vecq`
│   ├── cli.py             argparse entry-point
│   ├── metrics.py         cosine / L2 / inner-product distances
│   ├── flat.py            brute-force exact baseline (vectorised)
│   ├── hnsw.py            HNSW graph index
│   └── bench.py           synthetic data + recall vs QPS harness
├── tests/test_vecq.py     19 pytest cases
├── assets/                recall_vs_qps.png, build_time.png
├── pyproject.toml
└── README.md
```

## CLI

```text
vecq search <index.pkl> <queries.npy> [--k N] [--ef EF] [--metric NAME]
vecq bench  [--n N] [--dim D] [--k K] [--out DIR] [--ef-search CSV]
vecq demo   [--n N] [--dim D] [--k K]
vecq info   <index-file>
```

`vecq bench` builds an HNSW over a synthetic clustered dataset and prints a table of recall@10 vs QPS at four beam widths, plus two PNG charts in `--out`.

## Two indexes, one API

| Index | Search | Build | Use when |
| --- | --- | --- | --- |
| `FlatIndex` | exact, O(N·dim) | O(1) — just store the vectors | N ≲ 10k, or you need perfect recall. |
| `HNSW`     | approximate, O(log N) on a graph | O(N · efConstruction · M) | N ≳ 10k and you can tolerate a few percent recall loss. |

Both share the same `(distances, ids)` tuple shape so code that consumes search results doesn't need to care which one ran.

## Distance metrics

All three metrics return a *distance* (smaller = closer), so the same code path can sort and threshold across metrics without special cases.

- `cosine` — `1 − cos(a, b)`. The default for embedding search.
- `l2`     — Euclidean distance. Use when vectors are on a true length scale.
- `ip`     — Negative dot product. Equivalent to cosine after L2-normalisation.

## The recall-vs-throughput tradeoff

HNSW exposes one knob that matters: `ef_search`. Bigger beams walk more of the graph, get higher recall, but take more time per query. The chart above is the canonical curve: at `ef_search=20` we trade recall (0.89) for speed (213 qps); at `ef_search=100` we hit 1.00 recall at 70 qps. The right point depends entirely on your application's tolerance for "almost right" answers.

## Metadata filtering

Pass a predicate to `search()` and HNSW prunes non-matching nodes during the beam search:

```python
only_recent = lambda doc_id: int(doc_id.split("-")[1]) > 1990
distances, ids = index.search(query, k=10, filter_fn=only_recent)
```

The beam is enlarged to compensate for rejected candidates so `k` results still come back even with selective predicates.

## Persistence

```python
index.save("my_index.pkl")
restored = HNSW.load("my_index.pkl")
```

`save()` pickles a versioned payload: the vectors, the per-node neighbour lists, the entry point and the max layer. The on-disk format is independent of NumPy or pickle protocol versions at read time because we version the file.

## Why this implementation is interesting

A few decisions worth pointing at:

- **Beam search with two priority queues.** A min-heap of candidates and a max-heap of current results. The prune rule — pop the worst result if a candidate is farther than it — is what keeps search at O(log N) instead of O(N).
- **Batched distance computation.** The hot inner loop of `_search_layer` calls one NumPy operation per layer, not one per neighbour. Without batching, pure Python is too slow to be usable.
- **Diversity heuristic for neighbour selection.** When picking which new links to create during insert, candidates that are too close to an already-chosen neighbour are skipped, which keeps the graph navigable instead of degenerate.
- **Layer-aware neighbour caps.** Upper layers cap at `M` links per node; layer 0 (the dense base) caps at `2M`. That asymmetry is what makes the long jumps in the upper layers actually useful.

## When to reach for something else

vecq is a teaching and prototyping library. If you need:

- **million-scale production indexing**, look at [hnswlib](https://github.com/nmslib/hnswlib) or [Faiss](https://github.com/facebookresearch/faiss). They compile to C++ and are orders of magnitude faster.
- **GPU indexing**, look at [cuVS](https://github.com/rapidsai/cuvs).
- **server-managed indexes**, look at [Qdrant](https://github.com/qdrant/qdrant), [Milvus](https://github.com/milvus-io/milvus) or [Weaviate](https://github.com/weaviate/weaviate).

vecq is for when you want to *understand* what those tools are doing, or build a small in-process index for an application that doesn't need million-scale.

## Running the tests

```bash
git clone https://github.com/Luv-Goel/vecq
cd vecq
python -m pip install -e ".[dev]"
pytest -q
```

19 tests cover the metrics, both indexes, persistence round-trip, metadata filtering, and the benchmark harness.

## Continuous integration

A GitHub Actions workflow is configured in `.github/workflows/ci.yml` that runs `pytest` on Linux, macOS and Windows across Python 3.10–3.13. It wasn't included in the initial push because the publishing token lacks the `workflow` scope; see [`docs/CI.md`](docs/CI.md) for the matrix and the two-step process to enable it.

## License

MIT — see [`LICENSE`](LICENSE).