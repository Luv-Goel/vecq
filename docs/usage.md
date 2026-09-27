# Usage Guide

vecq is designed to be extremely easy to use while offering the same fundamental abstractions found in production systems like Faiss or Qdrant.

## Basic Vector Search

The core workflow involves creating an index, adding vectors, and querying them.

=== "HNSW (Approximate)"

    ```python
    import numpy as np
    from vecq import HNSW, HNSWConfig

    # 1. Generate some dummy data
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((10_000, 128)).astype(np.float64)
    # L2 normalization for cosine similarity
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # 2. Initialize the index
    config = HNSWConfig(M=16, ef_construction=200, ef_search=50, metric="cosine")
    index = HNSW(dim=128, config=config)

    # 3. Add vectors to the graph
    index.extend(vectors)

    # 4. Search!
    query = rng.standard_normal(128)
    query /= np.linalg.norm(query)
    distances, ids = index.search(query, k=5)

    print("Top 5 matches:")
    for dist, doc_id in zip(distances, ids):
        print(f"ID: {doc_id} | Distance: {dist:.4f}")
    ```

=== "Flat (Exact)"

    ```python
    import numpy as np
    from vecq import FlatIndex

    # 1. Generate some dummy data
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((10_000, 128)).astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # 2. Build the exact flat index
    index = FlatIndex.build(vectors, metric="cosine")

    # 3. Search! (Runs an exhaustive matrix multiplication)
    query = rng.standard_normal(128)
    query /= np.linalg.norm(query)
    distances, ids = index.search(query, k=5)

    print("Top 5 exact matches:")
    for dist, doc_id in zip(distances, ids):
        print(f"ID: {doc_id} | Distance: {dist:.4f}")
    ```

## External Identifiers

By default, vecq assigns a sequential integer ID to every inserted vector. However, you often want to map vectors to your own database IDs (e.g. UUIDs or strings). 

```python
docs = [
    {"id": "doc-a1b2", "vector": [...]},
    {"id": "doc-c3d4", "vector": [...]},
]

for doc in docs:
    index.add(doc["vector"], external_id=doc["id"])

# Search returns your custom string IDs
distances, ids = index.search(query, k=3)
# ids -> array(['doc-c3d4', ...])
```

## Metadata Filtering

Sometimes you only want to search over a subset of your data (e.g., "only search documents published after 2020"). vecq supports runtime metadata filtering using a predicate function.

```python
# A simple lambda that checks the ID (or looks up a dict)
only_recent = lambda doc_id: int(doc_id.split("-")[1]) > 2020

# The HNSW graph will automatically widen its beam search 
# to ensure it still finds `k` results that pass the filter!
distances, ids = index.search(query, k=10, filter_fn=only_recent)
```

!!! tip "Performance with Filters"
    If your filter rejects > 50% of the nodes in the graph, HNSW will automatically bump the internal `ef_search` size to compensate for the rejected candidates and maintain high recall.

## Saving and Loading

Both `HNSW` and `FlatIndex` support versioned serialization to disk. The on-disk format is independent of NumPy or pickle protocol versions at read time.

```python
# Save to disk
index.save("my_vectors.pkl")

# Load later (or in another process)
# vecq handles restoring the complex hierarchical graph structure automatically
restored_index = HNSW.load("my_vectors.pkl")
```
