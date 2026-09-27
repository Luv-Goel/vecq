# Contributing

Pull requests welcome. vecq is small on purpose.

## Reporting bugs

Open an issue with:

- A minimal reproduction (a short script that imports vecq and shows the problem).
- The NumPy version, Python version and platform.
- The exact error text or unexpected behaviour, including any printed recall/QPS numbers.

## Setting up locally

```bash
git clone https://github.com/Luv-Goel/vecq
cd vecq
python -m pip install -e ".[dev]"
pytest -q
```

The full suite must pass before a PR is reviewed.

## Style

- Pure Python with NumPy as the only runtime dependency. No torch, no sklearn, no scipy.
- Type hints throughout (`from __future__ import annotations`).
- Test new code with `pytest`. Every public function or method gets at least one test.
- Keep modules small. If `hnsw.py` exceeds ~400 lines, something belongs in its own module.

## Architecture notes

- All three distance metrics return a *distance* — smaller is closer. Don't introduce a similarity-direction metric; consumers rely on the invariant.
- The HNSW graph is mutated in place during `add()`. Calling `save()` serialises the whole graph; calling `load()` returns a fresh index. No thread safety today; do not share an index across threads.
- Batch distance computation in `_distances_batch` is what makes the search usable. If you add a new metric, extend the batched fast path there too — falling back to scalar Python defeats the point.
- Filtering widens the search beam automatically. If you add a filter mode that rejects > 50% of nodes, also add a note in the README about how to size `ef`.

## Commit messages

Imperative mood, 50 chars or less for the subject. Examples:

- `Add IP distance fast path to _distances_batch`
- `Fix HNSW edge case when target_level exceeds max_level`
- `Document recall-vs-throughput curve in docs/`