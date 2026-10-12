# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. This is a research project: most defects worth finding are about
**reproducibility** (a hard-coded hyperparameter that bypasses the configs, an
unseeded draw, a DVC stage that does not declare what it reads, a metric
written somewhere the pipeline does not look, statistics leaking from the
test split into training) or about **where code lives** (a loader or a
training loop re-implemented in a script or a notebook instead of reused from
the package), not style. Read "Boundaries", "Reuse before you write" and
"The contracts" in [`AGENTS.md`](AGENTS.md) first; this file is the
checklist and the report format.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import
order, unused imports); review what they cannot see.

### 1. Reuse and where code lives

- Every function, class or module the diff **adds** has been checked against
  [`docs/api/capabilities.md`](docs/api/capabilities.md). A loader, model,
  training loop, seeding helper or metric re-implemented in a script or a
  notebook is a **High** finding, with the existing name to use.
- Reusable logic is in the package (`src/<package>/`), with a test; scripts
  only read the config, call the package and write outputs; notebooks
  import from the package.
- The package does not import Hydra, DVC, a script or a notebook, and does
  not read a `DictConfig`; it takes plain arguments.
- A new public name is in the regenerated capability index
  (`make capabilities`).

### 2. Configs

- No hyperparameter, seed, split fraction, path or model choice written as
  a literal in a script or notebook: it is a key in `configs/`, read from
  `cfg`, with its default in the YAML.
- A new config group option provides every key the scripts read from that
  group; a renamed key is renamed everywhere it is read (scripts,
  `dvc.yaml` `params`, notebooks, docs).
- The hydra-zen entry point and the YAML configs still configure the same
  objects with the same defaults.
- Sweeps (`-m`) do not write every job's result to the same path.

### 3. The DVC pipeline

- Every stage lists as `deps` its script, the package modules it imports
  and the data it reads; as `params` the config keys it uses (keys that
  exist in the params file); as `outs` / `metrics` / `plots` exactly what
  the script writes.
- No output path contains a file git tracks; no data or result file is
  committed to git; `.dvc` pointers and `dvc.lock` are committed with the
  change that produced them.
- A new script that produces results is a stage, not a manual step.

### 4. Reproducibility and data hygiene

- One seed per run, from the config (`seed_everything`, or an explicit
  `np.random.default_rng(seed)` passed down); no unseeded draws, no
  global-state RNG calls inside the package.
- Splits are made once, before any fitting, from config fractions and the
  seed; normalisation statistics, vocabularies, imputers and feature
  selection are fitted on the training split only; nothing tuned on the
  test split.
- Order-dependent operations are deterministic (sorted file listings,
  stable sorts, `set` iteration not relied on); non-deterministic GPU
  kernels are called out.
- Metrics are written as JSON to the path `dvc.yaml` declares, with keys
  that `dvc metrics diff` and the results notebook read.
- Paths are `pathlib.Path`, built from the project root (or an argument),
  never from the current working directory Hydra may change.

### 5. Notebooks, docs and citation

- Notebooks are jupytext percent-format `.py` (in `notebooks/`) or marimo
  apps (in `marimo_notebooks/`); no `.ipynb` is added; a figure used by the
  docs is saved under `docs/images/<notebook_name>/` and committed with it.
- A number or figure quoted in the docs comes from a pipeline output or a
  notebook that reads one.
- A new page is in the `myst.yml` toc; references are BibTeX entries in
  `references.bib` cited with `[@key]`; `CITATION.cff` is updated when the
  authors, title or version change.

### 6. Tests

- New package code has tests in `tests/` against a hand computation, a
  closed form or a small fixture (`tmp_path` for files); no network, no
  real data, well under a second each.
- Incidental randomness is seeded; a test on a random quantity bounds it by
  its sampling distribution, with the bound's source in a comment.

### 7. Python, dependencies and security

- Type hints on every public function; `X | None`; specific exceptions
  with `raise … from …`; guard clauses over deep nesting; Google-style
  docstrings with the formula for a scientific computation (Unicode, e.g.
  `σ² = Σ(xᵢ − μ)² / N`).
- A new runtime dependency is in `pixi.toml` with `pixi.lock` regenerated
  (and in `pyproject.toml`'s `dependencies` when the package imports it);
  nothing is installed by hand.
- No credentials, tokens or DVC remote secrets in the diff (they belong in
  `.env` or `.dvc/config.local`, both gitignored).

---

## Research-Specific Checks

<!-- template-example: the examples use the example package (myproject); keep the patterns, rename the imports. -->

The snippets below run as written from the repo root in the pixi
environment, with `cfg` composed from `configs/`:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir


with initialize_config_dir(
    version_base=None, config_dir=str(Path("configs").resolve())
):
    cfg = compose(config_name="train", overrides=["training.epochs=3"])
```

### Settings come from the config

```python
from hydra.utils import instantiate

from myproject.models.baseline import BaselineModel
from myproject.trainers.trainer import Trainer

# ❌ Literals: a run cannot be reproduced or swept from the command line,
#    and DVC's params tracking never sees them
model = BaselineModel(hidden_size=128, num_layers=4)
trainer = Trainer(model=model, lr=0.01, epochs=50)

# ✅ Every value from cfg; override with `training.lr=0.01 model=<name>`
#    (each configs/model/<name>.yaml names its class in `_target_`)
model = instantiate({k: v for k, v in cfg.model.items() if k != "name"})
trainer = Trainer(model=model, lr=cfg.training.lr, epochs=cfg.training.epochs)
```

### One seed, from the config

```python
import numpy as np

x = np.arange(10.0)

# ❌ A different split on every run
idx = np.random.permutation(len(x))

# ✅ The same split for the same config
rng = np.random.default_rng(cfg.training.seed)
idx = rng.permutation(len(x))
```

### No leakage from the test split

```python
n_train = int(cfg.data.train_split * len(x))

# ❌ The mean and standard deviation include the test samples
x_std = (x - x.mean()) / x.std()
train, test = x_std[idx[:n_train]], x_std[idx[n_train:]]

# ✅ Split first; fit the statistics on the training split only
train, test = x[idx[:n_train]], x[idx[n_train:]]
mu, sigma = train.mean(), train.std()
train, test = (train - mu) / sigma, (test - mu) / sigma
```

### A stage declares what it reads and writes

```yaml
# ❌ DVC cannot tell when evaluate is stale, and does not track its output
evaluate:
  cmd: python scripts/evaluate.py

# ✅
evaluate:
  cmd: python scripts/evaluate.py
  deps:
    - scripts/evaluate.py
    - data/processed
    - results/metrics/train_metrics.json
  metrics:
    - results/metrics/eval_metrics.json:
        cache: false
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
- Don't nitpick what linters and formatters catch, or code the change did
  not touch
