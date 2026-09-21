"""Command-line interface for vecq.

  vecq search <index.npz> <queries.npy> [--k N] [--ef EF] [--metric NAME]
  vecq bench  [--n N] [--dim D] [--k K] [--out DIR]
  vecq demo   [--n N] [--dim D] [--k K]
  vecq info   <index-file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .bench import make_dataset, run_benchmark, plot_recall_vs_qps, plot_indexing_time, format_table
from .hnsw import HNSW, HNSWConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vecq", description="vector search from scratch")
    parser.add_argument("--version", action="version", version=f"vecq {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="query a saved HNSW index")
    p.add_argument("index", help="path to a .pkl file produced by HNSW.save")
    p.add_argument("queries", help="path to a .npy file of query vectors")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef", type=int, default=None, help="beam width during search")
    p.add_argument("--metric", choices=("cosine", "l2", "ip"), default="cosine")

    p = sub.add_parser("bench", help="run the recall-vs-QPS benchmark")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=100)
    p.add_argument("--ef-search", default="20,50,100,200", help="comma-separated ef_search values")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="assets", help="directory for the PNG charts")

    p = sub.add_parser("demo", help="build a tiny index, query it, print neighbours")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--k", type=int, default=5)

    p = sub.add_parser("info", help="print stats for a saved HNSW index")
    p.add_argument("index")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "search":
        return _search(args)
    if args.command == "bench":
        return _bench(args)
    if args.command == "demo":
        return _demo(args)
    if args.command == "info":
        return _info(args)
    return 1


def _search(args: argparse.Namespace) -> int:
    index = HNSW.load(args.index)
    queries = np.load(args.queries)
    if queries.ndim == 1:
        queries = queries[np.newaxis]
    for offset, query in enumerate(queries):
        distances, ids = index.search(query, k=args.k, ef=args.ef)
        print(f"# query {offset}")
        for rank, (distance, ident) in enumerate(zip(distances, ids), start=1):
            print(f"  {rank:>3}  id={ident!s:<20} distance={distance:.6f}")
    return 0


def _bench(args: argparse.Namespace) -> int:
    ef_search = [int(value) for value in args.ef_search.split(",") if value]
    vectors = make_dataset(args.n, args.dim, seed=args.seed)
    result = run_benchmark(
        vectors,
        k=args.k,
        ef_search_values=ef_search,
        n_queries=args.queries,
        M=args.m,
        ef_construction=args.ef_construction,
        metric=args.metric,
        seed=args.seed,
    )
    print(format_table(result))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chart1 = plot_recall_vs_qps(result, str(out / "recall_vs_qps.png"))
    chart2 = plot_indexing_time(result, str(out / "build_time.png"))
    print(f"\nWrote {chart1}\nWrote {chart2}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    vectors = make_dataset(args.n, args.dim)
    index = HNSW(args.dim, HNSWConfig(M=8, ef_construction=50, ef_search=20, metric="cosine"))
    index.extend(vectors)
    rng = np.random.default_rng(123)
    query = rng.standard_normal(args.dim)
    query /= np.linalg.norm(query)
    distances, ids = index.search(query, k=args.k)
    print(f"query:\n  {query.round(3).tolist()}")
    print(f"\nTop {args.k} neighbours:")
    for rank, (distance, ident) in enumerate(zip(distances, ids), start=1):
        print(f"  {rank:>3}  id={ident!s:<8} distance={distance:.4f}")
    return 0


def _info(args: argparse.Namespace) -> int:
    index = HNSW.load(args.index)
    for key, value in index.stats().items():
        print(f"{key:<18} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())