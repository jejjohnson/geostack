---
name: pre-pr-check
description: Run the research project's full pre-PR verification — tests, ruff lint and format on the whole repo, ty, the capability index, the DVC pipeline status for changed stages, the docs build, and the lockfiles — the way CI runs them. Use before committing, pushing or opening a pull request, and after any change to the package, configs, dvc.yaml, dependencies, notebooks or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why.

## Always

```bash
make test                                   # uv: every test, no coverage
uv run --group lint ruff check .            # entire repo: tests/, scripts/
uv run --group lint ruff format --check .
make typecheck                              # ty on the package
```

CI runs the same through pixi (`ci.yml`, `lint.yml`, `typecheck.yml`), on
Ubuntu and macOS for the tests; when pixi is installed, confirm with:

```bash
pixi run test                               # pytest -v with coverage (writes coverage.xml)
pixi run lint
pixi run typecheck
```

A test that needs Hydra or another pixi-only package starts with
`pytest.importorskip(...)`, so it skips under uv and runs under pixi: when
you touched configs or scripts, run `pixi run test`.

## When the package's public names, a layer's table or a submodule changed

```bash
make capabilities                                   # all four layers
pixi run pytest tests/test_capabilities.py -v -o addopts=
pixi run -e geomodels pytest tests/test_capabilities.py -v -o addopts=
```

…and commit `docs/api/capabilities*.md` with `pixi.lock`. Without
conda-forge, push the branch: `pixi-lock.yml` regenerates and commits both.

## When configs, scripts or `dvc.yaml` changed

```bash
pixi run dvc dag                  # the stage graph still parses and is connected
pixi run dvc status               # which stages are stale
pixi run dvc repro <stage>        # rerun the stages you changed
pixi run dvc metrics diff         # what moved, for the PR description
```

Commit `dvc.lock` (and `cache: false` metrics files) with the change; never
commit data or outputs. (The unadopted template ships no `dvc.lock`: it has
no data; delete the one a check run writes.) `dvc-check.yml` and `experiment-report.yml` run on
the PR; `reproducibility.yml` reruns everything weekly. The
`reproduce-results` skill lists the usual failures.

## When dependencies changed

`pixi lock` and commit `pixi.lock` (a runtime dependency belongs in
`pixi.toml`, and in `pyproject.toml`'s `dependencies` too when the package
imports it); `uv.lock` is gitignored, since `pixi.lock` is the lockfile of
record.

## When notebooks changed

```bash
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" notebooks/*.py
MPLBACKEND=Agg pixi run python notebooks/<file>.py
pixi run -e marimo marimo check marimo_notebooks/<file>.py
```

No `.ipynb` and no `__marimo__/` in `git status`; figures the docs use are
committed under `docs/images/`.

## When docs, `myst.yml`, `references.bib` or `CITATION.cff` changed

```bash
make docs                                   # read the ⛔️ / ⚠️ lines for your pages
uvx cffconvert --validate -i CITATION.cff   # what citation.yml runs
```

If the theme download is blocked, see the `write-up-results` skill.

## Before pushing

- `git status` shows no stray files (`.plans/`, `_build/`, `outputs/`,
  `multirun/`, `dvc_plots/`, generated `.ipynb`).
- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
