# Architecture

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

## Why this implementation is interesting

A few decisions worth pointing at:

- **Beam search with two priority queues.** A min-heap of candidates and a max-heap of current results. The prune rule — pop the worst result if a candidate is farther than it — is what keeps search at O(log N) instead of O(N).
- **Batched distance computation.** The hot inner loop of `_search_layer` calls one NumPy operation per layer, not one per neighbour. Without batching, pure Python is too slow to be usable.
- **Diversity heuristic for neighbour selection.** When picking which new links to create during insert, candidates that are too close to an already-chosen neighbour are skipped, which keeps the graph navigable instead of degenerate.
- **Layer-aware neighbour caps.** Upper layers cap at `M` links per node; layer 0 (the dense base) caps at `2M`. That asymmetry is what makes the long jumps in the upper layers actually useful.
