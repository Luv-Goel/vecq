# Welcome to vecq

A vector search engine from scratch — **HNSW approximate search** and **brute-force exact search** with cosine, L2 and inner-product distance metrics, metadata filtering, persistence, and a benchmark harness. Pure Python + NumPy.

![HNSW vs brute-force: recall vs throughput](../assets/recall_vs_qps.png)

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
