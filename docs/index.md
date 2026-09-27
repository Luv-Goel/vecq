# vecq 🚀

> A vector search engine from scratch in pure Python + NumPy.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Luv-Goel/vecq/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/Luv-Goel/vecq/actions)

**vecq** provides both **HNSW approximate search** and **brute-force exact search** with cosine, L2 and inner-product distance metrics, metadata filtering, persistence, and a built-in benchmark harness.

## Why vecq?

Approximate nearest-neighbour search underpins every modern retrieval system that touches embeddings — RAG pipelines, semantic search, recommendation engines, and image search. Hierarchical Navigable Small World (HNSW) is the algorithm behind most of them. 

I wanted to know how it actually works, so I wrote it from scratch.

vecq is a clean-room implementation you can read in one sitting. It's small enough to fit in your head, fast enough to use for prototyping (thanks to batched NumPy operations), and accurate enough to compare against production libraries on recall-vs-throughput tradeoffs.

---

## Quickstart

```bash
git clone https://github.com/Luv-Goel/vecq
cd vecq
python -m pip install -e .
```

Run the built-in CLI demo to see it in action:
```bash
python -m vecq demo --n 1000 --dim 64 --k 5
```

**Next up:** Head over to the [Usage Guide](usage.md) to see how to use vecq in your Python code!
