---
name: add-notebook
description: Add or update a notebook in the research project — a jupytext percent-format .py notebook in notebooks/ (converted to .ipynb only locally) or a marimo app in marimo_notebooks/ — that imports from the package, reads pipeline outputs, is seeded, and saves the figures the docs use under docs/images/<notebook>/. Use when asked to write an exploration, EDA, results, figure or analysis notebook, or to update one after an API or pipeline change.
---

# Add a notebook

Read contract 5 ("Notebooks") in `AGENTS.md` first. Two kinds, both plain
`.py` files; `.ipynb` files are never committed (`*.ipynb` is gitignored).

| Kind | Where | For | Edit with |
|---|---|---|---|
| jupytext percent format | `notebooks/NN_<topic>.py` | Linear analyses: EDA, results, the figures in the write-up | `pixi run -e jupyterlab lab` (open the `.ipynb` made by `pixi run notebooks-to-ipynb`, or pair them) |
| marimo app | `marimo_notebooks/NN_<topic>.py` | Interactive, reactive exploration (sliders, linked views) | `pixi run -e marimo marimo-edit` |

## 1. Make sure it does not exist yet

`ls notebooks marimo_notebooks`: extend `01_eda.py` (or
`01_exploration.py`) when the analysis belongs there. Reusable code a
notebook needs goes in the package (`docs/api/capabilities.md`), not in a
cell.

## 2. A jupytext notebook

<!-- template-example: the header and paths follow the example's notebooks; keep the format. -->

Copy the YAML header and layout of `notebooks/01_eda.py`:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # <Title>
#
# One paragraph: the question this notebook answers and the outputs it reads.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geostack.utils.reproducibility import seed_everything


seed_everything(42)
rng = np.random.default_rng(42)
```

- Cells are `# %%` (code) and `# %% [markdown]` (text, every line starting
  with `# `).
- Import from the package; read pipeline outputs (`results/metrics/*.json`,
  `results/figures/*.csv`, `data/processed/`) by paths relative to the repo
  root — run the notebook from there. Read the keys the producing script
  actually writes (open the JSON once rather than assuming its layout).
- Seed every random draw (`np.random.default_rng(seed)`).
- A figure the docs use is saved, then shown:

  ```python
  out = Path("docs/images/03_ablation")
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / "loss_vs_lr.png", dpi=150, bbox_inches="tight")
  plt.show()
  ```

  and committed together with the notebook (the docs do not render
  notebooks; they show the saved figures).

## 3. A marimo notebook

Copy `marimo_notebooks/01_exploration.py`: a PEP 723 header listing what
the app imports (so `uv run <file>` can run it on its own), then
`app = marimo.App(...)` and one `@app.cell` function per cell, each
returning the names later cells use. Edit it in the browser
(`pixi run -e marimo marimo edit marimo_notebooks/<file>.py`), which keeps
the file valid; marimo's `__marimo__/` session cache next to it is not
committed.

## 4. Lint and parse

ruff skips `notebooks/` and `marimo_notebooks/` (`exclude` in
`pyproject.toml`); CI (`notebooks.yml`) only checks that
`notebooks/*.py` parse, so run them yourself.

## 5. Verify

```bash
# jupytext: it parses (what CI checks), runs top to bottom, and converts
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" notebooks/<file>.py
MPLBACKEND=Agg pixi run python notebooks/<file>.py
pixi run jupytext --to notebook notebooks/<file>.py      # local .ipynb; do not commit it

# marimo: lint the app, then run it as a script
pixi run -e marimo marimo check marimo_notebooks/<file>.py
MPLBACKEND=Agg pixi run -e marimo python marimo_notebooks/<file>.py
```

Delete the generated `.ipynb` / `__marimo__/` before committing if `git
status` shows them, and commit any figures under `docs/images/`.
