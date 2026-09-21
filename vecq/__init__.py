"""vecq — vector search engine from scratch.

A pure Python + NumPy library that implements:

  * Flat (brute-force) exact search for ground-truth comparison.
  * HNSW (Hierarchical Navigable Small World) for fast approximate search.
  * Multiple distance metrics: cosine, L2, inner product.
  * Metadata filtering during search.
  * Persistent indexes via pickle.

NumPy is the only runtime dependency.
"""

from .hnsw import HNSW, HNSWConfig
from .flat import FlatIndex
from .metrics import METRICS

__version__ = "1.0.0"

__all__ = ["HNSW", "HNSWConfig", "FlatIndex", "METRICS", "__version__"]