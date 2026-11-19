---
name: add-experiment
description: Add an experiment to the research project — a named Hydra config (or a new config key / group option), a DVC stage when it produces new outputs, its metrics and plots, and a test that the config composes — so the run is reproducible with `dvc repro` and comparable with `dvc metrics diff` / `dvc exp show`. Use when asked to add, set up, configure or sweep an experiment, ablation, baseline run, hyperparameter search or a new pipeline step (evaluation, analysis) in configs/, scripts/ and dvc.yaml.
---

# Add an experiment

Read contracts 2 to 4 ("Configs", "The pipeline", "Results") in
`AGENTS.md` first; this is the step-by-step.

**Where this repo stands.** geostack has no package of its own, and until
the first experiment lands it has no `configs/` directory, no entry script
and no pipeline stage (`dvc.yaml` is `stages: {}`); what exists today are
notebooks run by hand. Look before you assume a file exists
(`ls configs scripts`, `cat dvc.yaml`, the `[tasks]` in `pixi.toml`). A
setting can be overridden or swept only once it is a config key read by a
script that a stage runs: if it is a literal in a notebook, say what has to
exist first. Below, `<entry>` is the entry script and config of the
experiment (`scripts/<entry>.py`, `configs/<entry>.yaml`).

## 1. Make sure it does not exist yet

- `ls configs/*/` and the entry config: the setting, group option or
  named experiment may already be there; a variant of an existing run is a
  command-line override (`training.lr=0.01 model=<name>`), not new code and
  not a new config file.
- `pixi run dvc dag` and `dvc.yaml`: the step may already be a stage.
- `pixi run dvc exp show` lists experiments already run from this commit.
- Reusable logic (a metric, a loss, an evaluation loop) is not written in
  the script: search the capability indexes, `docs/api/capabilities*.md`,
  and use what is there; if it is missing it goes upstream, into the
  library that owns it (the `add-model` and `add-dataset` skills cover
  models and data). Say which index you searched and what you found.

## 2. Pick the kind

| The experiment is… | Write | Run it with |
|---|---|---|
| A different value of an existing setting | nothing (an override), or a new default in the entry config | `pixi run dvc exp run -S 'configs/<entry>.yaml:training.lr=0.01'` |
| A new setting | a key with its default in the entry config (or the group file), read from `cfg` in the script | the same |
| A named combination worth keeping (an ablation, the paper's main run) | `configs/experiment/<name>.yaml` (below) | `+experiment=<name>` in the stage's `cmd`, then `pixi run dvc repro` |
| A sweep to compare | nothing new | `pixi run dvc exp run -S 'configs/<entry>.yaml:training.lr=0.01'` per point, then `pixi run dvc exp show` |
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
stage's command (`cmd: python scripts/<entry>.py +experiment=<name>`); the
stage depends on all of `configs/`.

A run whose result will be compared or reported goes through DVC
(`dvc exp run`, `dvc repro`), never the script by hand. The two results are
then compared from tracked outputs: `pixi run dvc exp show`,
`dvc metrics diff`, `dvc params diff`. Give no result numbers, and predict
none, for a run you did not make. To keep one point of a sweep,
`pixi run dvc exp apply <name>`, then commit `dvc.lock` and the
`cache: false` metrics file: a number in the docs comes from that commit.

Hydra multirun (`python scripts/<entry>.py -m training.lr=0.001,0.01`) is fine for a
quick look, but every job writes the same `results/metrics/` files, so only
the last survives: use `dvc exp run` (one tracked experiment per point) for
anything you will compare or report.

## 3. Configs

- Every value the run depends on is a config key with its default in the
  YAML; the script reads it from `cfg`, never a literal.
- A group option (`configs/model/<name>.yaml`, `configs/data/<name>.yaml`)
  provides every key the scripts read from that group.
- If a hydra-zen entry point configures the same objects, give its
  `builds(...)` / `make_config(...)` the same defaults.

## 4. The script (only for a new step)

- `@hydra.main(version_base=None, config_path="../configs", config_name="<entry>")`;
- seed first, from the config, if anything is random
  (`rng = np.random.default_rng(cfg.seed)`, passed down);
- call the upstream libraries — no model, loader, metric or loop defined in
  the script as something another script would want;
- write outputs under the project root
  (`Path(get_original_cwd()) / "results" / ...`), to exactly the paths the
  stage declares: metrics as a flat JSON dict of numbers in
  `results/metrics/<step>_metrics.json`, plots as CSV / JSON in
  `results/figures/`, tables in `results/tables/`;
- add a pixi task in `pixi.toml` (`[tasks]`) if people will run it by hand
  (there is none for experiments yet), and write the whole changed
  `pixi.toml`, not a note about it;
- it reads its inputs from what an earlier stage writes. If nothing in the
  repo produces an input yet, say so plainly: the stage cannot run until
  that stage exists.

## 5. The DVC stage

```yaml
  evaluate:
    cmd: python scripts/evaluate.py
    deps:
      - scripts/evaluate.py
      - references/geoml/kernellib   # the submodule of every library it imports
      - data/processed
      - results/metrics/train_metrics.json
    params:
      - configs/<entry>.yaml:
          - training                  # keys that exist in the params file
    metrics:
      - results/metrics/eval_metrics.json:
          cache: false
    plots:
      - results/figures/eval_curve.csv
```

- A stage whose script reads a config declares the keys it reads under
  `params`, so `dvc params diff` and `dvc exp show` see them.
- With no local package, the code a stage imports is upstream: depend on
  the library's submodule folder (or on `pixi.lock`), so a moved pin reruns
  the stage.
- `params` names keys of the file: a Hydra defaults group (`model`,
  `data`) is not a key of the entry config, and DVC stops with
  "Parameters 'model' are missing". Depend on the group's file instead
  (`deps: [configs/model/baseline.yaml]`, or the whole `configs/`).
- No `outs` path may contain a git-tracked file (a `.gitkeep`), or DVC
  refuses to run the stage.
- Metrics are a flat JSON dict of scalars (final or summary values);
  per-epoch or per-step values go in the plots file, which is what
  notebooks plot from.
- A `cache: false` metrics file is meant to be committed to git, so
  `dvc metrics diff` can compare it with `HEAD`. `.gitignore` keeps
  `results/metrics/*.json` (`!results/metrics/*.json`); for a metrics file
  elsewhere, check `git check-ignore -v <file>` and add a `!` exception if
  a rule hides it.

## 6. Tests

- Upstream code the experiment needs is tested upstream.
- The config composes and provides what the scripts read: add a case to
  `tests/test_configs.py` (create it with the first config; Hydra is in the
  pixi environment only, so the module starts with
  `pytest.importorskip("hydra")` and runs under pixi):

```python
CONFIGS = Path(__file__).parents[1] / "configs"
EXPERIMENTS = sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))


def _compose(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name="<entry>", overrides=overrides)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes(experiment: str) -> None:
    cfg = _compose([f"+experiment={experiment}"])
    for key in ("lr", "epochs"):
        assert key in cfg.training
    instantiate({k: v for k, v in cfg.model.items() if k != "name"})
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

## 8. Report

Say what you added and what you had to create because the repo did not
have it, which commands you ran and which you did not run and why, what
the user runs next, and where the results will be read from. No numbers
for runs that were not made.
