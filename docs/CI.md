# Continuous integration

A GitHub Actions workflow is configured in `.github/workflows/ci.yml`. The
file is present in the working tree but is not pushed to `main` because
the OAuth scope used to publish the repository does not include `workflow`,
which GitHub requires for both direct git pushes and Contents API writes
that create or modify files under `.github/workflows/`.

## What the workflow does

When enabled, it runs `pytest -q` against vecq on:

- **Operating systems:** Ubuntu, macOS, Windows.
- **Python versions:** 3.10, 3.11, 3.12, 3.13.

That's a 12-cell matrix. A second job byte-compiles every source file with
`python -m compileall` as a cheap syntax sanity check.

## How to enable it

1. Open the repository on GitHub → **Settings → Actions → General** and
   ensure Actions are allowed.
2. Commit `.github/workflows/ci.yml` from the working tree to `main` using
   a token that has the `workflow` scope (a fine-grained PAT with
   `workflows: write` works), or push it through the GitHub web editor.
3. The next push to `main` (or any pull request) will trigger the matrix.

Until then, the local suite passes:

```
$ pytest -q
...................                                                      [100%]
```