---
name: pre-pr-check
description: Run the research project's full pre-PR verification — tests, ruff lint and format on the whole repo, ty, the capability index, the DVC pipeline status for changed stages, the docs build, and the lockfiles — the way CI runs them. Use before committing, pushing or opening a pull request, and after any change to the package, configs, dvc.yaml, dependencies, notebooks or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why. When asked what
to run for a change, list the "Always" checks first, in full, then the
sections the change triggers, then "Before pushing"; say which sections do
not apply and why, and what each check proves.

## Always

```bash
make test                                   # pixi run pytest -v, every test, no coverage
uv run --group lint ruff check .            # entire repo: tests/, scripts/
uv run --group lint ruff format --check .
make typecheck                              # ty on scripts/ (there is no package)
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

## When a layer's table in `scripts/capabilities.py` or a submodule changed

```bash
pixi lock                                           # the libraries install from the submodules
make capabilities                                   # all four layers
pixi run pytest tests/test_capabilities.py -v -o addopts=              # GeoStack
pixi run -e <layer> pytest tests/test_capabilities.py -v -o addopts=   # numerics, geoml or geomodels
make dependency-graph                               # README.md and docs/api/dependency_graph.md
pixi run pytest tests/test_dependency_graph.py -v -o addopts=
```

- A library that is new to the stack also gets its layer and tier in
  `TIERS` in `scripts/dependency_graph.py`, or the graph test fails.
- Commit the submodule pointer, `pixi.lock`, `docs/api/capabilities*.md`,
  `README.md` and `docs/api/dependency_graph.md` together. Without
  conda-forge, push the branch: `pixi-lock.yml` regenerates and commits the
  lock and the indexes.

## When a skill, a plugin or a contract card changed

```bash
pixi run pytest tests/test_skill_tree.py -v -o addopts=   # budgets, Builds-on blocks, cards, the marketplace
make skill-tree                                     # only if a skill was added, renamed or its description changed
make skill-shelves                                  # only if a skill-shelf page's library changed
pixi run pytest tests/test_skill_shelf.py tests/test_shelf_check.py -v -o addopts=
```

This covers `plugins/geostack/`, `skill_tree/cards/` and
`.claude-plugin/marketplace.json`. A change to a skill's text alone needs
the first line, and no DVC, capability or notebook check.

## When configs, scripts or `dvc.yaml` changed

```bash
pixi run dvc dag                  # the stage graph still parses and is connected
pixi run dvc status               # which stages are stale
pixi run dvc repro <stage>        # rerun the stages you changed
pixi run dvc metrics diff         # what moved, for the PR description
```

Commit `dvc.lock` (and `cache: false` metrics files) with the change; never
commit data or outputs. (This repo has no stages and no `dvc.lock` yet, so
these apply from the first stage on.) `dvc-check.yml` and `experiment-report.yml` run on
the PR; `reproducibility.yml` reruns everything weekly. The
`reproduce-results` skill lists the usual failures.

## When dependencies changed

`pixi lock` and commit `pixi.lock` (a runtime dependency belongs in
`pixi.toml`); `uv.lock` is gitignored, since `pixi.lock` is the lockfile of
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

- Read the ⛔️ and ⚠️ lines for your pages: `make docs` finishes even with
  them. A citation key that is not in `references.bib` is only a ⚠️.
- A new page is in the `toc` of `myst.yml`, or it is not built.
- `cffconvert` only when `CITATION.cff` changed.
- A number or a figure on a results page comes from a committed pipeline
  output, not from a session: check where the page's numbers come from, or
  ask.
- If the theme download is blocked, see the `write-up-results` skill.

## Before pushing

- `git status` shows no stray files (`.plans/`, `_build/`, `outputs/`,
  `multirun/`, `dvc_plots/`, generated `.ipynb`).
- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
