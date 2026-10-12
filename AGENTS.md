# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

> **Template note.** This repository is a GitHub template for research
> projects. Passages that describe the bundled example (the `myproject`
> package, its configs, its DVC stages, its notebooks) start with an HTML
> comment `<!-- template-example: … -->`; after adopting the template, rewrite each
> of them from your own code (the `adopt-template` skill walks through
> the whole adoption). List them with
> `grep -rnE '(<!--|#) template-example:' --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.pixi .`.
> Everything without the marker is a rule for every project made from the
> template.

## What this repo is

<!-- template-example: describes the example project; rewrite from your code. -->

A reproducible research project: an installable Python package holds the
reusable code, thin scripts run it, Hydra configs hold every setting, a DVC
pipeline ties data, code, configs and results together, notebooks explore
and report, and MyST docs write it up. Experiments, notebooks and the docs
are built on the package, so **the package is the project's library of
primitives**: a loader, a model or a training step written once there is
reused everywhere, and anything a script or a notebook needs twice moves
into it.

| Path | Owns |
|---|---|
| `src/myproject/data/loading.py` | `load_raw_data(data_dir)`, `load_processed_data(data_dir)` (placeholders: return dicts, raise `FileNotFoundError` for a missing directory) |
| `src/myproject/models/baseline.py` | `BaselineModel(hidden_size, num_layers)` with `forward(x)` |
| `src/myproject/models/transformer.py` | `TransformerModel(d_model, nhead, num_layers)` with `forward(x)` (a placeholder) |
| `src/myproject/trainers/trainer.py` | `Trainer(model, lr, epochs).train(train_data, val_data=None)` → `{"train_loss": [...], "val_loss": [...]}` |
| `src/myproject/utils/reproducibility.py` | `seed_everything(seed)`: seeds `random`, NumPy and, when installed, PyTorch (deterministic cuDNN) |
| `scripts/` | Entry points: `preprocess.py`, `train.py` (`@hydra.main` over `configs/train.yaml`), `train_zen.py` (the same models and settings with hydra-zen configs; prints the history, writes no pipeline outputs), `evaluate.py`, `convert_notebooks.py` (jupytext → `.ipynb`), `capabilities.py` (the capability index) |
| `configs/` | Hydra configs: `train.yaml` (defaults `model: baseline`, `data: small`; `training.*`, `logging.*`), `model/{baseline,transformer}.yaml` (each names its class in `_target_`), `data/{small,full}.yaml` |
| `dvc.yaml` | The pipeline: `preprocess` (`data/raw` → `data/processed`) → `train` (final losses → metrics `results/metrics/train_metrics.json`; per-epoch losses → plot `results/figures/loss_curve.csv`) |
| `data/{raw,processed,external}/`, `results/{metrics,figures,tables}/` | DVC-managed data and outputs: gitignored except the `.gitkeep` files, `.dvc` pointers and the `cache: false` metrics JSON; `data/processed/` is the `preprocess` stage's output and is created by it |
| `notebooks/` | Jupytext percent-format `.py` notebooks (`01_eda.py`, `02_results.py`) |
| `marimo_notebooks/` | marimo notebooks (`01_exploration.py`) |
| `docs/`, `myst.yml`, `references.bib` | MyST documentation (notes, tutorials, API pages, capability index) and its bibliography |
| `CITATION.cff` | How to cite the project |

`scripts/evaluate.py` writes `results/metrics/eval_metrics.json` but is not
a DVC stage yet; `configs/data/*.yaml` and `training.batch_size` /
`val_interval` are not read by any script yet.

### Boundaries

These hold for every project made from the template:

- **Dependencies point one way: scripts and notebooks → package.** The
  package never imports a script, Hydra, DVC or a notebook. It takes plain
  arguments (numbers, paths, arrays), not a `DictConfig`, so it can be
  tested and reused without a config.
- **Side effects live in scripts.** Reading configs, choosing paths, writing
  metrics and figures happen in `scripts/`; the package computes. A
  function that a script and a notebook both need belongs in the package.
- **Data and results never go into git.** They are DVC outputs or `dvc add`
  pointers; `.gitignore` keeps `data/*` and `results/*` out.
- **No GeoML-stack library is a dependency of the template.** A project
  that builds on one (gaussx, pyrox, xtremax, …) lists it in
  `UPSTREAM` in `scripts/capabilities.py` so its API joins the capability
  index.

## What it is built on

Each tool brings a rule. Breaking one runs fine once on your machine and
fails on the next run, the next machine or in CI.

| Tool | Used for | The rule it brings |
|---|---|---|
| **pixi** (`pixi.toml`, `pixi.lock`) | The runtime and CI environment: Python, NumPy, SciPy, xarray, netCDF4, matplotlib, DVC, hydra-core, hydra-zen, OmegaConf (conda-forge first, PyPI second); extra environments `docs`, `jupyterlab`, `marimo` | CI's test, lint, type-check, DVC and docs workflows run through `pixi run`. A runtime dependency goes into `pixi.toml` and the lock is regenerated (`pixi lock`, commit `pixi.lock`); one the package imports also goes into `dependencies` in `pyproject.toml`, so the package installs on its own. Never `pip install` / `conda install` into an environment by hand. |
| **uv** (`pyproject.toml`, `Makefile`) | The local developer loop: pytest, ruff, ty, pre-commit, the MyST CLI | `pyproject.toml` declares only the package's own runtime dependencies (NumPy), so the uv environment holds the package and the tooling but not Hydra or DVC: run tests, lint, types and docs with `make` / `uv run`, experiments with `pixi run`. `uv.lock` is gitignored: `pixi.lock` is the lockfile of record. Always `uv run`, never the system Python. |
| **Hydra** + **OmegaConf** (`configs/`, `scripts/train.py`) | Every setting of a run, composed from config groups and overridden on the command line | No hard-coded hyperparameter, seed, split or path in a script; read it from `cfg`. Outputs the pipeline tracks go to fixed project-root paths (`get_original_cwd()`), not into Hydra's run directory. |
| **hydra-zen** (`scripts/train_zen.py`) | Configs generated from Python signatures (`builds`, `make_config`, `ZenStore`) | The structured config is built from the class it configures (`builds(TheClass, populate_full_signature=True)`), so the two cannot drift apart. |
| **DVC** (`dvc.yaml`, `.dvc/`, `.dvcignore`) | Data versioning and the pipeline from raw data to metrics | Every stage declares everything it reads (`deps`, `params`) and everything it writes (`outs`, `metrics`, `plots`); commit `dvc.lock` and `.dvc` pointer files, never the data. |
| **NumPy** / `random` (optionally PyTorch) | Numerics and random draws | Seed once per run, from the config, with the package's seeding helper (`seed_everything(cfg.training.seed)` in the example); in new code prefer an explicit `np.random.default_rng(seed)` passed down. |
| **jupytext** / **marimo** | Notebooks as plain `.py` files | No `.ipynb` is committed (`*.ipynb` is gitignored); notebooks import from the package instead of defining reusable code. |
| **MyST** (`mystmd`, `myst.yml`) | Documentation and the write-up, with citations from `references.bib` | Pages are MyST Markdown listed in the `myst.yml` toc; cite with `[@key]`. |
| **pytest**, **ruff**, **ty** | Tests, lint and format (line length 88), type checks | See "Before every commit". |

## Reuse before you write

Before writing a loader, a model, a training step, a metric or a helper —
in the package, a script or a notebook — find out whether it exists:

1. **Search the capability index.**
   [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every
   public name in the package, once, with a one-line summary. It is
   generated (`make capabilities`) and checked by
   `tests/test_capabilities.py`, so it is current.
2. **Search the scripts and configs**: an entry point may already do it
   (`scripts/`), and a setting may already exist (`configs/`).
3. **If it is missing, add it to the package** in the subpackage that owns
   the concept (data loading, models, training, utilities), with a test,
   not inline in the one script or notebook that needs it today.
4. **One object, one name.** Two modules never export the same name for
   different things (`tests/test_capabilities.py`); a deliberate exception
   goes into `ALLOWED_SHARED_NAMES` in `scripts/capabilities.py` with its
   reason.

<!-- template-example: the names below are the example's; rewrite for your package. -->

| You are about to write… | Use instead |
|---|---|
| `random.seed(…)`, `np.random.seed(…)`, `torch.manual_seed(…)` at the top of a script | `myproject.utils.reproducibility.seed_everything(cfg.training.seed)` |
| File reading from `data/raw` or `data/processed` in a script or notebook | `load_raw_data`, `load_processed_data` (extend them) |
| An epoch loop in a script | `Trainer.train` |
| A model class in a script or notebook | `src/myproject/models/` (`BaselineModel` as the pattern) |
| A literal learning rate, epoch count, seed or split fraction | a key in `configs/` (`training.lr`, `training.seed`, `configs/data/*.yaml`), overridden on the command line |
| A config dataclass written by hand | hydra-zen `builds(TheClass, populate_full_signature=True)` |
| A shell script that runs the steps in order | a stage in `dvc.yaml`, run with `dvc repro` |
| A loop calling `jupytext` / `nbconvert` to make `.ipynb` files | `scripts/convert_notebooks.py` (`pixi run notebooks-to-ipynb`) |
| `os.path.join`, string paths | `pathlib.Path` |
| A reference typed into a page | an entry in `references.bib`, cited with `[@key]` |

## The contracts

The step-by-step for each is a skill (see "Recipes").

### 1. The package: pure, typed, tested

- Reusable code lives in `src/<package>/`, one concept per subpackage; the
  public functions and classes have type hints and Google-style docstrings
  and appear in the capability index.
- Functions take plain arguments (paths as `str | Path`, numbers, arrays)
  and return values; they do not read configs, environment variables or
  global state, and do not write files unless that is their job and the
  path is an argument.
- Randomness comes from a seed or generator the caller passes, or from
  `seed_everything` called once by the entry point.
- Every public function or class has a test in `tests/`.

### 2. Configs: every setting in `configs/`

- `configs/train.yaml` is the entry config; alternatives are config groups
  (`configs/model/`, `configs/data/`) chosen with `model=<name>` /
  `data=<name>` and composed through its `defaults` list.
- A script reads every value from `cfg`; a new hyperparameter is a new key,
  with its default in the YAML, never a literal in Python.
- A config group option must provide every key the script reads from that
  group (in the example, `train.py` builds `cfg.model` with
  `hydra.utils.instantiate`, so every `configs/model/*.yaml` names its class
  in `_target_` and lists that class's constructor arguments;
  `tests/test_configs.py` builds every option).
- Sweeps are Hydra multirun (`-m training.lr=0.001,0.01`); a sweep whose
  jobs write to the same fixed output path overwrites itself, so give each
  job its own output or use DVC experiments.
- The hydra-zen path (`scripts/train_zen.py`) generates its configs from
  signatures; keep the two entry points configuring the same objects.

### 3. The pipeline: DVC stages declare everything

- Each stage in `dvc.yaml` has a `cmd` and lists as `deps` the script, the
  package code it imports and the data it reads; as `params` the config
  keys it uses (keys that exist in the params file — a Hydra defaults group
  such as `model` is not a key of `configs/train.yaml`; depend on the group
  file instead); as `outs` what it writes. An output path must not contain
  files git tracks: DVC refuses to reproduce such a stage (no `.gitkeep`
  inside an output directory; the stage's script creates the directory).
- Metrics are flat JSON dicts of scalars (final or summary values) under
  `results/metrics/`, declared with `cache: false`; per-epoch or per-step
  values are plots (CSV or JSON) under `results/figures/`; the script writes
  exactly the paths the stage declares, and notebooks read those shapes.
- Raw data enters with `dvc add` (commit the `.dvc` pointer) and leaves
  with `dvc push`; `dvc.lock` is committed after `dvc repro`. The template
  itself ships no `dvc.lock` (it has no data and no remote, so the lock
  would pin placeholder outputs nobody can pull); commit one after your
  first `dvc repro`.
- What DVC expects in git must not be gitignored: the `.dvc` pointers and
  the `cache: false` metrics files (without them in `HEAD`,
  `dvc metrics diff` has nothing to compare). `git check-ignore -v <path>`
  shows the rule that hides a file; keep the template's
  `!data/raw/*.dvc` and `!results/metrics/*.json` exceptions when you
  change `.gitignore`.

### 4. Results: regenerable from a commit

- A number or figure in the docs or a paper comes from a pipeline output
  (or a notebook that reads one), never from a one-off session.
- Train / validation / test splits are made once, from config fractions and
  a seed, before any fitting; statistics used to transform data are fitted
  on the training split only.

### 5. Notebooks

- `notebooks/*.py` are jupytext percent-format scripts: a YAML header, then
  `# %%` / `# %% [markdown]` cells. `.ipynb` files are generated locally
  (`pixi run notebooks-to-ipynb`) and never committed.
- `marimo_notebooks/*.py` are marimo apps (`app = marimo.App(...)`,
  `@app.cell` functions) with a PEP 723 `# /// script` header listing their
  dependencies.
- Notebooks import from the package and read pipeline outputs; figures for
  the docs are saved under `docs/images/<notebook_name>/` and committed with
  the notebook.
- ruff skips both directories (`exclude` in `pyproject.toml`); CI only
  checks that `notebooks/*.py` parse.

### 6. Docs and citation

- Pages are MyST Markdown under `docs/`, each listed in the `myst.yml` toc;
  the bibliography is `references.bib`; `CITATION.cff` says how to cite the
  project and is validated in CI when it changes.

## What enforces them

<!-- template-example: the test rows describe the example's tests; rewrite them. -->

| Check | Enforces |
|---|---|
| `tests/test_data.py` | The loaders return the documented keys |
| `tests/test_models.py` | `BaselineModel` and `TransformerModel` defaults, custom parameters, `repr`, `forward` |
| `tests/test_configs.py` | Every `model=` option composes and builds as `train.py` builds it; `train_zen.py` offers the same options and defaults; `train.py` writes scalar metrics and a per-epoch plots CSV (skips without Hydra, i.e. outside pixi) |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name bound to two objects |
| ruff (`make lint`), ty (`make typecheck`) | Lint (`E F I UP B SIM RUF`) and types on `src/` |
| `.pre-commit-config.yaml` (`make precommit`) | ruff, ruff-format, whitespace, YAML / JSON syntax, files over 5 MB |
| `notebooks.yml` | `notebooks/*.py` parse; warns on committed `.ipynb` |
| `dvc-check.yml` | `dvc status` and `dvc dag` run on PRs touching the pipeline |
| `experiment-report.yml` | A PR comment with `dvc metrics diff`, `dvc params diff`, `dvc status` |
| `reproducibility.yml` | The whole pipeline (`dvc repro`) weekly and on demand |
| `citation.yml` | `CITATION.cff` validates |
| `conventional-commits.yml` | PR titles are Conventional Commits with a lowercase subject |

Nothing tests `Trainer`, `seed_everything`, `preprocess.py` or
`evaluate.py` yet, and `tests/test_configs.py` covers the `model` group
only: review the rest by hand.

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them), and two read-only review checklists in
`.claude/agents/`:

| Job | Recipe |
|---|---|
| Turn a fresh copy of the template into your project | `adopt-template` |
| Add an experiment: a config, a DVC stage, metrics, a sweep | `add-experiment` |
| Add a dataset: DVC-tracked raw data, a loader, a data config, splits | `add-dataset` |
| Add a model: a class, a config-group option, the training wiring | `add-model` |
| Add or update a jupytext or marimo notebook | `add-notebook` |
| Reproduce the results, or debug a pipeline that does not reproduce | `reproduce-results` |
| Write up methods and results in the MyST docs, cite, update `CITATION.cff` | `write-up-results` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `myproject-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `reproducibility-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `make gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

A research project is not a library other projects build on, so the
template ships no downstream plugin or `llms.txt`; the skills above are for
work inside the project.

## Working in the repo

Two environment managers share the repo. uv (via `make`) is the local
developer loop; pixi is what CI and the experiments run in.

```bash
make install              # uv sync --all-groups + pre-commit hooks
make test                 # uv run pytest -v -o addopts=   (no coverage)
make test-cov             # pytest with the coverage report (pyproject addopts)
make lint                 # ruff check .   (entire repo)
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check src/myproject
make precommit            # pre-commit on every file
make capabilities         # regenerate docs/api/capabilities.md
make docs                 # myst build --html  (into _build/html)
make docs-serve           # myst start (live preview)
make help                 # every target
```

```bash
pixi install                          # the default environment (-e <env> for another)
pixi run test                         # pytest -v with coverage (what CI runs)
pixi run lint                         # ruff check .      (what CI runs)
pixi run typecheck                    # ty check src/     (what CI runs)
pixi run preprocess                   # also: train, train-zen, evaluate
pixi run train training.lr=0.01 model=baseline   # Hydra overrides
pixi run dvc repro                    # the whole pipeline (dvc repro <stage>: one stage + upstream)
pixi run dvc status                   # what changed since dvc.lock
pixi run dvc dag                      # the stage graph
pixi run dvc add data/raw/<file>      # track a raw file (commit the .dvc pointer)
pixi run dvc push                     # upload data (dvc pull downloads); needs a
                                      # remote: dvc remote add -d <name> <url>
pixi run dvc metrics diff             # compare metrics with the last commit
pixi run notebooks-to-ipynb           # notebooks/*.py → .ipynb (local only)
pixi run -e docs docs-build           # what pages.yml runs
pixi run -e jupyterlab lab
pixi run -e marimo marimo-edit
```

`pixi.toml` pins `python` below 3.14: Hydra 1.3 does not run on Python
3.14 (`ValueError: badly formed help string` from argparse). Lift the pin
(and re-lock) only once the Hydra release in `pixi.lock` supports it.

Run one test with `uv run pytest tests/test_models.py::test_baseline_model_init -v -o addopts=`.

### Test tiers

- No markers are registered: every test runs everywhere. CI ("CI",
  `ci.yml`) runs `pixi run test` (`pytest -v` with the coverage `addopts`)
  on Ubuntu and macOS, uploads `coverage.xml` and gates nothing on it.
- `filterwarnings = ["error", "ignore::DeprecationWarning"]` turns any
  other warning into a failure; `xfail_strict = true`.
- Keep tests fast and offline. A test that needs data, a network or minutes
  of compute gets a marker: register it under `markers` in
  `[tool.pytest.ini_options]` first, deselect it in the `test` task, and
  run it in a separate workflow.
- The pipeline itself is checked by `reproducibility.yml` (`dvc repro`),
  not by pytest.

### Tests that assert on random draws

- **Incidental randomness** (any draw would do): a fixed seed
  (`np.random.default_rng(0)`, `seed_everything(0)`).
- **Behaviour under test** (a sampler, a stochastic training step): bound
  the statistic by its own sampling distribution (a number of standard
  errors), not a tolerance tuned on one draw, and say in a comment where
  the bound came from.

### Before every commit

All of these must pass, from the repo root:

1. `make test` (or `pixi run test`): zero failures.
2. `uv run --group lint ruff check .`: the **entire** repo, including
   `tests/` and `scripts/` (notebooks are excluded by the ruff config).
3. `uv run --group lint ruff format --check .`
4. `make typecheck` (or `pixi run typecheck`).
5. After changing a public name in the package: `make capabilities`.
6. After changing `dvc.yaml`, a stage's script or its configs:
   `pixi run dvc status` and `pixi run dvc dag`, and `pixi run dvc repro`
   for the stages you touched (commit `dvc.lock`).
7. After changing a dependency: `pixi lock` and commit `pixi.lock`.
8. After changing `docs/`, `myst.yml` or `references.bib`: `make docs`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; if something is
   unclear, stop and ask; push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions, no configurability
   nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change;
   remove only what your change made unused, and mention (don't delete)
   unrelated dead code.
4. **Goal-driven.** Turn the task into a check (a failing test, a
   reproduced bug, a metric to match) and loop until it passes; for
   multi-step work state the plan with a check per step.

Also: Python 3.12+, `from __future__ import annotations` in every module,
type hints on every public function, `X | None` over `Optional`,
`pathlib.Path` for filesystem work, specific exceptions with
`raise … from …`, ruff with line length 88.

## Git, commits and pull requests

- **Never** push to or merge into `main` / `master` unless explicitly told
  to ("push to main", "merge to main"). Work on a feature branch and commit
  locally; never run `git push` unless asked. "Merge the branch" means push
  the feature branch, not merge into `main`. Confirm before any action that
  affects a shared branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/)
  (`<type>[optional scope]: <description>`) with a lowercase subject;
  `conventional-commits.yml` validates PR titles. Types: `feat`, `fix`,
  `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`,
  `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please (`release-please-config.json`: one
  Python package at `.`, tags `<package-name>-vX.Y.Z`, the version bumped in
  `pyproject.toml`); don't bump versions by hand. A release is a natural
  point to archive results (Zenodo DOI, `CITATION.cff`).
- **Never replace or remove an existing PR title or description.** An
  agent called for a small follow-up tends to write a fresh description
  scoped to its own work and silently discard the rest: read the existing
  description first, treat it as the base, and only append checklist items
  or update their status.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the
  label taxonomy and epic model in [`CONTRIBUTING.md`](CONTRIBUTING.md)
  (`make gh-labels`, `gh-sub`, `gh-block`, `gh-show`).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

MyST Markdown built by the `mystmd` CLI from `myst.yml` at the repo root
(book theme); `pages.yml` builds it (`pixi run -e docs docs-build`) and
deploys `_build/html` to GitHub Pages on every push to `main`. No workflow
builds the docs on a pull request, so run `make docs` yourself.

- **Pages** live under `docs/` (`notes/` for background and methods,
  `tutorials/`, `api/`), each listed in the `toc` of `myst.yml`; a page
  that is not in the toc is not built into the site.
- **Citations**: add the BibTeX entry to `references.bib` and cite it with
  `[@key]` (or `{cite:t}` / `{cite:p}`).
- **Figures** from notebooks go to `docs/images/<notebook_name>/`, committed
  together with the notebook that makes them.
- **Notebooks** are not rendered into the site; the docs show their saved
  figures.
- **API pages**: `docs/api/capabilities.md` is generated
  (`make capabilities`); `docs/api/reference.md` is hand-written.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never
committed); track work in GitHub issues (`docs/templates/wave-backlog.md`
drafts a whole wave of them).
