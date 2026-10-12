---
name: add-experiment
description: Add an experiment to the research project — a named Hydra config (or a new config key / group option), a DVC stage when it produces new outputs, its metrics and plots, and a test that the config composes — so the run is reproducible with `dvc repro` and comparable with `dvc metrics diff` / `dvc exp show`. Use when asked to add, set up, configure or sweep an experiment, ablation, baseline run, hyperparameter search or a new pipeline step (evaluation, analysis) in configs/, scripts/ and dvc.yaml.
---

# Add an experiment

<!-- template-example: example-specific paths (configs/train.yaml, scripts/train.py, myproject); keep the procedure, rewrite the names after adoption. -->

Read contracts 2 to 4 ("Configs", "The pipeline", "Results") in
`AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- `ls configs/*/` and `configs/train.yaml`: the setting, group option or
  named experiment may already be there; a variant of an existing run is a
  command-line override (`training.lr=0.01 model=baseline`), not new code.
- `pixi run dvc dag` and `dvc.yaml`: the step may already be a stage.
- `pixi run dvc exp show` lists experiments already run from this commit.
- Reusable logic (a metric, a loss, an evaluation loop) goes in the
  package first: search `docs/api/capabilities.md` (the `add-model` and
  `add-dataset` skills cover models and data).

## 2. Pick the kind

| The experiment is… | Write | Run it with |
|---|---|---|
| A different value of an existing setting | nothing (an override), or a new default in `configs/train.yaml` | `pixi run train training.lr=0.01` |
| A new setting | a key with its default in `configs/train.yaml` (or the group file), read from `cfg` in the script | the same |
| A named combination worth keeping (an ablation, the paper's main run) | `configs/experiment/<name>.yaml` (below) | `pixi run train +experiment=<name>` |
| A sweep to compare | nothing new | `pixi run dvc exp run -S 'configs/train.yaml:training.lr=0.01'` per point, then `pixi run dvc exp show` |
| A new step with new outputs (evaluation, analysis, a figure) | `scripts/<step>.py` and a `dvc.yaml` stage (steps 4 and 5) | `pixi run dvc repro <stage>` |

A named experiment config overrides the entry config from the global
package, and may switch groups. `configs/experiment/fast_lr.yaml` (the
`# @package _global_` line must be the file's first line, or the group
overrides fail to compose):

```yaml
# @package _global_
defaults:
  - override /model: baseline
  - override /data: small

training:
  lr: 1.0e-2
  epochs: 5
```

To make a named experiment the pipeline's run, put the override in the
stage's command (`cmd: python scripts/train.py +experiment=<name>`); the
`train` stage already depends on all of `configs/`.

Hydra multirun (`pixi run train -m training.lr=0.001,0.01`) is fine for a
quick look, but every job writes the same `results/metrics/` files, so only
the last survives: use `dvc exp run` (one tracked experiment per point) for
anything you will compare or report.

## 3. Configs

- Every value the run depends on is a config key with its default in the
  YAML; the script reads it from `cfg`, never a literal.
- A group option (`configs/model/<name>.yaml`, `configs/data/<name>.yaml`)
  provides every key the scripts read from that group.
- If `scripts/train_zen.py` configures the same objects, give its
  `builds(...)` / `make_config(...)` the same defaults.

## 4. The script (only for a new step)

Follow `scripts/train.py`:

- `@hydra.main(version_base=None, config_path="../configs", config_name="train")`
  (or a new entry config next to `train.yaml`);
- `seed_everything(cfg.training.seed)` first, if anything is random;
- call the package — no model, loader or loop defined in the script;
- write outputs under the project root
  (`Path(get_original_cwd()) / "results" / ...`), to exactly the paths the
  stage declares: metrics as a flat JSON dict of numbers in
  `results/metrics/<step>_metrics.json`, plots as CSV / JSON in
  `results/figures/`, tables in `results/tables/`;
- add a pixi task in `pixi.toml` (`[tasks]`) if people will run it by hand.

## 5. The DVC stage

```yaml
  evaluate:
    cmd: python scripts/evaluate.py
    deps:
      - scripts/evaluate.py
      - src/myproject/models          # every package directory it imports
      - data/processed
      - results/metrics/train_metrics.json
    params:
      - configs/train.yaml:
          - training                  # keys that exist in the params file
    metrics:
      - results/metrics/eval_metrics.json:
          cache: false
    plots:
      - results/figures/eval_curve.csv
```

- `params` names keys of the file: a Hydra defaults group (`model`,
  `data`) is not a key of `configs/train.yaml`, and DVC stops with
  "Parameters 'model' are missing". Depend on the group's file instead
  (`deps: [configs/model/baseline.yaml]`, or the whole `configs/`).
- No `outs` path may contain a git-tracked file (a `.gitkeep`), or DVC
  refuses to run the stage.
- Metrics are a flat JSON dict of scalars (final or summary values);
  per-epoch or per-step values go in the plots file, which is what
  notebooks plot from (`scripts/train.py`'s `write_outputs` is the
  pattern).
- A `cache: false` metrics file is meant to be committed to git, so
  `dvc metrics diff` can compare it with `HEAD`. `.gitignore` keeps
  `results/metrics/*.json` (`!results/metrics/*.json`); for a metrics file
  elsewhere, check `git check-ignore -v <file>` and add a `!` exception if
  a rule hides it.

## 6. Tests

- Package code the experiment needs: unit tests in `tests/`, as usual.
- The config composes and provides what the scripts read: add a case to
  `tests/test_configs.py`, next to the `model=` cases, reusing its
  `_compose` and `_load_script` helpers (Hydra is in the pixi environment
  only, so the module skips elsewhere):

```python
EXPERIMENTS = sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes(experiment: str) -> None:
    cfg = _compose([f"+experiment={experiment}"])
    for key in ("lr", "epochs", "seed"):
        assert key in cfg.training
    _load_script("train").build_model(cfg.model)  # as train.py builds it
```

## 7. Verify

```bash
pixi run pytest tests/test_configs.py -v -o addopts=   # the composition test
pixi run dvc dag                                       # the stage is wired in
pixi run dvc repro <stage>                             # runs it and its upstream stages
pixi run dvc metrics show                              # the metrics landed where declared
pixi run dvc params diff                               # what changed vs HEAD
```

Then commit the configs, the script, `dvc.yaml`, `dvc.lock` and the
`cache: false` metrics files, and run the `pre-pr-check` skill. If
`dvc repro` fails, the `reproduce-results` skill lists the usual causes.
