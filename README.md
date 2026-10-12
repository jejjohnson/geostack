# GeoStack

[![CI](https://github.com/jejjohnson/geostack/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/geostack/actions/workflows/ci.yml)
[![Docs](https://github.com/jejjohnson/geostack/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/geostack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![DVC](https://img.shields.io/badge/data%20versioned%20with-DVC-945DD6)](https://dvc.org)

> A JAX-native research software stack for geophysical modeling, inference and data assimilation.

## Features

- 🐍 **Python 3.12+** with modern type annotations
- 📦 **[Pixi](https://pixi.sh)** for reproducible conda-forge environments across platforms
- 📊 **[DVC](https://dvc.org)** for data versioning and ML pipeline management
- ⚙️ **[Hydra](https://hydra.cc)** + **[hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/)** for type-safe configuration management
- 📓 **[JupyterLab](https://jupyterlab.readthedocs.io)** with LSP, Git integration, and MyST rendering
- 📝 **[Jupytext](https://jupytext.readthedocs.io)** — notebooks stored as diff-friendly `.py` files, convert to `.ipynb` on demand
- 🌊 **[Marimo](https://marimo.io)** reactive notebook environment
- 📚 **[MyST-MD](https://mystmd.org)** for publication-quality documentation
- 🔍 **[Ruff](https://docs.astral.sh/ruff/)** for fast linting and formatting
- 🔒 **Pre-commit hooks** for code quality enforcement
- 🚀 **15 GitHub Actions workflows** for CI/CD, reproducibility, and automation
- 🖥️ **Cross-platform** support (Linux x86-64, macOS ARM64)
- 📝 **[CITATION.cff](https://citation-file-format.github.io/)** for academic citations
- 🔬 **Hydra multirun** sweeps for hyperparameter optimization
- 🤖 **Agent-ready**: [`AGENTS.md`](AGENTS.md) holds the rules every coding agent follows (with the example-specific passages marked for rewriting), [`CODE_REVIEW.md`](CODE_REVIEW.md) the review checklist, [`docs/api/capabilities.md`](docs/api/capabilities.md) indexes every public name in the upstream libraries (`make capabilities`), and `.claude/` holds step-by-step skills (adopt the template, add an experiment / dataset / model / notebook, reproduce and write up results) and two read-only reviewers

## Project Structure

```
geostack/
├── .github/workflows/    # 15 GitHub Actions workflows
├── .devcontainer/        # VS Code / Codespaces dev container
├── data/                 # Data directories (DVC-managed)
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/                 # MyST documentation
├── marimo_notebooks/     # Marimo reactive notebooks
├── notebooks/            # Jupyter notebooks
├── results/              # Experiment results (DVC-managed)
├── scripts/              # Entry point scripts (no local package: see upstream libraries)
└── tests/                # Test suite
```

## Quick Start

### Prerequisites

Install [Pixi](https://pixi.sh):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Installation

```bash
git clone https://github.com/jejjohnson/geostack
cd geostack
pixi install
```

## Environments

| Environment | Features | Use Case |
|-------------|----------|----------|
| `default` | dev | Testing, linting, training |
| `docs` | docs | Building MyST documentation |
| `jupyterlab` | dev + jupyterlab | Interactive notebooks |
| `marimo` | dev + marimo | Reactive notebooks |

```bash
# Activate specific environment
pixi run -e jupyterlab lab
pixi run -e marimo marimo-edit
pixi run -e docs docs-build
```

## Configuration Management

Experiments use [Hydra](https://hydra.cc) and [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/): every setting of a run lives in `configs/` (added with the first experiment) and is overridable on the command line. See the `add-experiment` skill.

## Experiment Tracking

This project uses [DVC](https://dvc.org) for data and experiment tracking.

```bash
# Add data to DVC
dvc add data/raw/dataset.csv

# Run the full pipeline
dvc repro

# Check pipeline status
dvc status

# View pipeline DAG
dvc dag

# Compare metrics across experiments
dvc metrics diff
```

The DVC pipeline is defined in `dvc.yaml` (no stages yet).

## Documentation

Documentation is built with [MyST-MD](https://mystmd.org) using the book theme.

```bash
# Serve locally with live reload
pixi run -e docs docs-serve

# Build static HTML
pixi run -e docs docs-build
```

Docs are automatically deployed to GitHub Pages on every push to `main`.

## Notebook Environments

Notebooks are stored as [Jupytext](https://jupytext.readthedocs.io) percent-format
`.py` scripts — not `.ipynb` files. This keeps the repository clean, diff-friendly,
and free of committed cell outputs. Generated `.ipynb` files are gitignored.

### Converting to .ipynb

```bash
# Convert all notebooks/ scripts to .ipynb
pixi run notebooks-to-ipynb

# Convert a single file
pixi run jupytext --to notebook notebooks/01_eda.py

# Keep .py and .ipynb in sync while editing (paired mode)
pixi run -e jupyterlab jupytext --set-formats py:percent,ipynb notebooks/01_eda.py
```

### JupyterLab

Full-featured JupyterLab with LSP, Git integration, MyST rendering, and spell checking:

```bash
pixi run -e jupyterlab lab
```

### Marimo

Reactive, reproducible notebooks in pure Python:

```bash
pixi run -e marimo marimo-edit
```

Marimo notebooks in `marimo_notebooks/` are also stored as `.py` files,
making them diff-friendly and importable as regular Python modules.

## CI/CD Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | push/PR | pytest on ubuntu + macos |
| `lint.yml` | push/PR | Ruff linting |
| `typecheck.yml` | push/PR | ty type checking |
| `pages.yml` | push to main | Build + deploy MyST docs |
| `dvc-check.yml` | DVC file changes | Validate DVC pipeline |
| `notebooks.yml` | notebook changes | Validate Jupytext .py notebooks |
| `reproducibility.yml` | weekly schedule | Full `dvc repro` |
| `experiment-report.yml` | PR | DVC metrics diff comment |
| `citation.yml` | CITATION.cff changes | Validate citation file |
| `pixi-update.yml` | monthly schedule | Update pixi lockfile |
| `codeql.yml` | push/PR/schedule | Security scanning |
| `conventional-commits.yml` | PR | Validate PR title |
| `label-pr.yml` | PR | Auto-label PRs |
| `pre-commit-autoupdate.yml` | weekly schedule | Update pre-commit hooks |
| `release-please.yml` | push to main | Open release PRs, tag releases, update `CHANGELOG.md` and versions |

## Academic

### Citation

If you use this software, please cite it using the metadata in `CITATION.cff`:

```bibtex
@software{johnson2026geostack,
  author = {Johnson, J. Emmanuel},
  title  = {GeoStack},
  year   = {2026},
  url    = {https://github.com/jejjohnson/geostack},
}
```

### References

Add BibTeX references to `references.bib`. They are automatically available
in MyST documentation and Jupyter notebooks with the `jupyterlab-myst` extension.

### Zenodo

Connect your GitHub repository to [Zenodo](https://zenodo.org) for automatic
DOI assignment on releases.

## Acknowledgments

This template was inspired by:
- [jejjohnson/pypackage_template](https://github.com/jejjohnson/pypackage_template) — library-focused Python package template
- [DrivenData Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Hydra documentation](https://hydra.cc/docs/intro/)
- [DVC documentation](https://dvc.org/doc)
