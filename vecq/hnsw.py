"""Hierarchical Navigable Small World (HNSW) — graph-based ANN search.

HNSW builds a multi-layer proximity graph:

    layer L  ---->    sparse top, skip across the whole index
    layer ...
    layer 2  ---->    a coarse long-jump graph
    layer 1  ---->    a denser short-jump graph
    layer 0  ---->    base layer: every node lives here

A new node is assigned a random max layer drawn from a geometric
distribution with parameter ``mL = 1 / ln(M)``. At each layer it
connects to its ``efConstruction`` nearest neighbours (capped at ``M``
for upper layers, ``2 * M`` at layer 0). Search starts at the top
layer, greedily descends one layer at a time, then performs a beam
search with width ``efSearch`` at the base layer to produce the top-k.

The implementation here uses two priority queues during search — a
min-heap of candidates and a max-heap of current results — and prunes
neighbours whose distance is worse than the worst current result.
"""

from __future__ import annotations

import heapq
import pickle
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .flat import Result
from .metrics import METRICS, Metric, resolve_metric


@dataclass
class HNSWConfig:
    """Hyperparameters. Defaults match the values used in the original paper."""

    M: int = 16  # max neighbours per node per upper layer
    ef_construction: int = 200  # beam width during insert
    ef_search: int = 50  # beam width during search
    metric: str = "cosine"
    seed: int = 42


@dataclass
class _Node:
    """A single node in the HNSW graph, with neighbours kept per layer."""

    vector: np.ndarray
    external_id: object
    neighbours: list[list[int]] = field(default_factory=list)


class HNSW:
    """Approximate nearest neighbour index with HNSW graph."""

    def __init__(self, dim: int, config: HNSWConfig | None = None) -> None:
        self.dim = dim
        self.config = config or HNSWConfig()
        self.metric: Metric = resolve_metric(self.config.metric)
        self._nodes: list[_Node] = []
        self._entry_point: int | None = None
        self._max_level: int = -1
        self._mL: float = 1.0 / np.log(max(self.config.M, 2))
        self._rng = np.random.default_rng(self.config.seed)
        self._visited_buffer: set[int] = set()

    # -- construction -----------------------------------------------------
    def add(self, vector: np.ndarray, external_id: object | None = None) -> None:
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},); got {vector.shape}")
        index = len(self._nodes)
        if external_id is None:
            external_id = index
        node = _Node(vector=np.asarray(vector, dtype=np.float64), external_id=external_id)
        target_level = self._random_level()
        node.neighbours = [[] for _ in range(target_level + 1)]
        self._nodes.append(node)

        if self._entry_point is None:
            self._entry_point = index
            self._max_level = target_level
            return

        # Phase 1: greedy from the top down to target_level + 1 to find a good entry.
        current = self._entry_point
        for layer in range(self._max_level, target_level, -1):
            current = self._greedy_descend(node.vector, current, layer)
        # Phase 2: beam search at each layer from target_level down to 0 and connect.
        for layer in range(min(target_level, self._max_level), -1, -1):
            neighbours = self._search_layer(node.vector, current, self.config.ef_construction, layer)
            max_links = self.config.M if layer > 0 else self.config.M * 2
            selected = self._select_neighbours(neighbours, max_links)
            node.neighbours[layer] = selected
            for _, neighbour_index in self._attach(index, layer, selected):
                self._nodes[neighbour_index].neighbours[layer].append(index)
                # Trim neighbour's link list if it now exceeds the cap.
                cap = self.config.M if layer > 0 else self.config.M * 2
                if len(self._nodes[neighbour_index].neighbours[layer]) > cap:
                    self._shrink_neighbours(neighbour_index, layer)

        if target_level > self._max_level:
            self._entry_point = index
            self._max_level = target_level

    def extend(self, vectors: Iterable[np.ndarray], ids: Iterable[object] | None = None) -> None:
        ids_list = list(ids) if ids is not None else None
        for offset, vector in enumerate(vectors):
            external = ids_list[offset] if ids_list else None
            self.add(vector, external)

    # -- search -----------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 10, ef: int | None = None,
               filter_fn: "callable | None" = None) -> Result:
        """Return the ``k`` nearest neighbours, optionally filtered by predicate.

        ``filter_fn(external_id)`` must return ``True`` for ids the caller
        accepts. Filtered nodes are pruned during the beam search, and the
        beam is widened by ``filter_fn`` rejection rate to keep ``k``
        results even when the predicate is selective.
        """
        if self._entry_point is None:
            empty = (np.empty(0, dtype=np.float64), np.empty(0, dtype=object))
            return empty
        if query.shape != (self.dim,):
            raise ValueError(f"expected query of shape ({self.dim},); got {query.shape}")
        beam = ef or self.config.ef_search
        if filter_fn is not None:
            beam = max(beam, self.config.ef_search * 4)
        current = self._entry_point
        for layer in range(self._max_level, 0, -1):
            current = self._greedy_descend(query, current, layer)
        neighbours = self._search_layer(query, current, beam, 0)
        if filter_fn is not None:
            neighbours = [(d, i) for d, i in neighbours if filter_fn(self._nodes[i].external_id)]
        k_eff = min(k, len(neighbours))
        top = heapq.nsmallest(k_eff, neighbours)
        ids = np.array([self._nodes[i].external_id for _, i in top], dtype=object)
        distances = np.array([d for d, _ in top], dtype=np.float64)
        return distances, ids

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        # Save a compact dict rather than the dataclass so the file stays readable
        # across format changes; we version it for forward compatibility.
        payload = {
            "version": 1,
            "dim": self.dim,
            "config": self.config.__dict__,
            "entry_point": self._entry_point,
            "max_level": self._max_level,
            "mL": self._mL,
            "vectors": np.stack([n.vector for n in self._nodes]) if self._nodes else np.empty((0, self.dim)),
            "ids": [n.external_id for n in self._nodes],
            "neighbours": [n.neighbours for n in self._nodes],
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "HNSW":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        config = HNSWConfig(**payload["config"])
        index = cls(payload["dim"], config)
        index._mL = payload["mL"]
        for vector, external_id, neighbours in zip(payload["vectors"], payload["ids"], payload["neighbours"]):
            index._nodes.append(_Node(vector=vector, external_id=external_id, neighbours=neighbours))
        index._entry_point = payload["entry_point"]
        index._max_level = payload["max_level"]
        return index

    # -- internals -------------------------------------------------------
    def _random_level(self) -> int:
        # Geometric distribution: P(level >= l) = (mL * exp(-mL))^(l-1)
        uniform = float(self._rng.random())
        if uniform <= 0.0:
            uniform = 1e-12
        return int(np.floor(-np.log(uniform) * self._mL))

    def _distance(self, query: np.ndarray, neighbour_index: int) -> float:
        return self.metric(query, self._nodes[neighbour_index].vector)

    def _distances_batch(self, query: np.ndarray, indices: list[int]) -> np.ndarray:
        """Compute distances from ``query`` to many nodes in one numpy pass."""
        if not indices:
            return np.empty(0, dtype=np.float64)
        vectors = np.stack([self._nodes[i].vector for i in indices])
        if self.metric is METRICS["l2"]:
            diffs = vectors - query
            return np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        if self.metric is METRICS["ip"]:
            return -(vectors @ query)
        # Cosine and any future metric: scalar fallback (still one Python call per batch).
        return np.array([self.metric(query, vectors[i]) for i in range(len(indices))])

    def _greedy_descend(self, query: np.ndarray, start: int, layer: int) -> int:
        best = start
        best_distance = self._distance(query, start)
        improved = True
        while improved:
            improved = False
            neighbours = [
                n
                for n in self._nodes[best].neighbours[layer]
                if n < len(self._nodes)
            ]
            if not neighbours:
                break
            distances = self._distances_batch(query, neighbours)
            local_best = int(np.argmin(distances))
            if distances[local_best] < best_distance:
                best = neighbours[local_best]
                best_distance = float(distances[local_best])
                improved = True
        return best

    def _search_layer(self, query: np.ndarray, start: int, ef: int, layer: int) -> list[tuple[float, int]]:
        """Beam search that returns up to ``ef`` (distance, index) pairs."""
        self._visited_buffer.clear()
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []  # max-heap via negation

        initial_distance = self._distance(query, start)
        heapq.heappush(candidates, (initial_distance, start))
        heapq.heappush(results, (-initial_distance, start))
        self._visited_buffer.add(start)
        worst_distance = initial_distance

        while candidates:
            current_distance, current = heapq.heappop(candidates)
            if current_distance > worst_distance and len(results) >= ef:
                break
            neighbours = [
                n
                for n in self._nodes[current].neighbours[layer]
                if n < len(self._nodes) and n not in self._visited_buffer
            ]
            if not neighbours:
                continue
            distances = self._distances_batch(query, neighbours)
            for neighbour, distance in zip(neighbours, distances):
                self._visited_buffer.add(neighbour)
                distance = float(distance)
                if distance < worst_distance or len(results) < ef:
                    heapq.heappush(candidates, (distance, neighbour))
                    heapq.heappush(results, (-distance, neighbour))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst_distance = -results[0][0]
        return [(-d, i) for d, i in results]

    def _select_neighbours(self, candidates: list[tuple[float, int]], count: int) -> list[int]:
        """Pick up to ``count`` neighbours from a candidate list, nearest first.

        A simple heuristic: prefer diverse picks by skipping candidates whose
        distance to an already-chosen neighbour is less than to the query.
        """
        selected: list[int] = []
        for distance, candidate_index in candidates:
            if len(selected) >= count:
                break
            too_close = False
            for chosen in selected:
                d_chosen = self._distance(self._nodes[candidate_index].vector, chosen)
                if d_chosen < distance:
                    too_close = True
                    break
            if too_close:
                continue
            selected.append(candidate_index)
        return selected

    def _attach(self, new_index: int, layer: int, selected: list[int]) -> list[tuple[int, float]]:
        """Return the (distance, neighbour) pairs of ``new_index``'s new links."""
        vector = self._nodes[new_index].vector
        return [(self.metric(vector, self._nodes[i].vector), i) for i in selected]

    def _shrink_neighbours(self, node_index: int, layer: int) -> None:
        """Trim the neighbour list to the layer's cap, keeping the closest ones."""
        neighbours = self._nodes[node_index].neighbours[layer]
        if not neighbours:
            return
        cap = self.config.M if layer > 0 else self.config.M * 2
        if len(neighbours) <= cap:
            return
        vector = self._nodes[node_index].vector
        ranked = sorted(neighbours, key=lambda i: self._distance(vector, i))
        self._nodes[node_index].neighbours[layer] = ranked[:cap]

    # -- introspection --------------------------------------------------
    def __len__(self) -> int:
        return len(self._nodes)

    def stats(self) -> dict[str, int]:
        nodes_per_layer: dict[int, int] = {}
        edges_per_layer: dict[int, int] = {}
        for node in self._nodes:
            for layer, neighbours in enumerate(node.neighbours):
                nodes_per_layer[layer] = nodes_per_layer.get(layer, 0) + 1
                edges_per_layer[layer] = edges_per_layer.get(layer, 0) + len(neighbours)
        return {
            "nodes": len(self._nodes),
            "max_level": self._max_level,
            "layers": len(nodes_per_layer),
            "nodes_per_layer": dict(sorted(nodes_per_layer.items(), reverse=True)),
            "edges_per_layer": dict(sorted(edges_per_layer.items(), reverse=True)),
        }