# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the project
layout, the tools and their rules, "reuse before you write", the contracts
for the package, configs, the DVC pipeline, results, notebooks and docs,
what enforces them, commands, the pre-commit checklist, git and PR rules).

The essentials, in case you only read this file:

- This is a research project template. Passages marked
  `<!-- template-example: … -->` in the agent files describe the bundled example
  package (`src/myproject/`) and are rewritten after adoption.
- Reusable code lives in the package; `scripts/` are thin Hydra entry
  points; notebooks import from the package. Search
  [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a
  loader, model, training step or helper.
- Every setting lives in `configs/` (Hydra; hydra-zen in
  `scripts/train_zen.py`), never as a literal in a script.
- Data and results are DVC-managed, never committed to git; every
  `dvc.yaml` stage declares its `deps`, `params`, `outs` and `metrics`;
  commit `dvc.lock`.
- Notebooks are jupytext percent-format `.py` files in `notebooks/` and
  marimo apps in `marimo_notebooks/`; no `.ipynb` is committed.
- Two environments: `make` / `uv run` for tests, lint, types and docs;
  `pixi run` for experiments and DVC (CI runs everything through pixi).
  Never `pip install` or `conda install` by hand.
- Propose a test with every feature or fix (write it first, then make it
  pass), and a concrete change with every review suggestion.
- Before committing, from the repo root: `make test`,
  `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`,
  `make typecheck`; `make capabilities` after a public API change.
- Step-by-step recipes (adopt the template, add an experiment / dataset /
  model / notebook, reproduce results, write them up; pre-PR check;
  review) are plain Markdown in `.claude/skills/<name>/SKILL.md`, and the
  two review checklists (`reuse-reviewer`, `reproducibility-reviewer`) in
  `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
