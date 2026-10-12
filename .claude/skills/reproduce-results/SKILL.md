---
name: reproduce-results
description: Reproduce the research project's results from a commit — install the pixi environment, pull the DVC-tracked data, run the pipeline with `dvc repro`, and check the metrics and plots against the committed ones — and diagnose a pipeline that does not reproduce. Use when asked to run, re-run or reproduce the pipeline or an experiment, regenerate results or figures, check that results are reproducible, or debug a failing `dvc repro` / reproducibility.yml run.
---

# Reproduce the results

Read contracts 3 and 4 in `AGENTS.md` first. The pipeline is `dvc.yaml`;
`dvc.lock` records the hashes of every dependency, parameter and output of
the last committed run, so a clean reproduction changes nothing.

## 1. Environment and data

```bash
pixi install                      # the environment pixi.lock pins (what CI uses)
pixi run dvc pull                 # the data and cached outputs, from the DVC remote
pixi run dvc status               # what differs from dvc.lock (nothing, after a pull)
pixi run dvc dag                  # the stages and their order
```

`dvc pull` needs a configured remote (`pixi run dvc remote list`; add one
with `pixi run dvc remote add -d <name> <url>`, credentials in
`.dvc/config.local`). Without a remote, `dvc repro` rebuilds everything
from the raw data in `data/raw/`.

## 2. Run

```bash
pixi run dvc repro                # every stage whose deps or params changed
pixi run dvc repro <stage>        # one stage and the stages it depends on
pixi run dvc repro --force        # rerun everything, to prove it reproduces
```

Never run the scripts by hand to produce reported results: only `dvc repro`
records what produced them in `dvc.lock`.

## 3. Compare

```bash
pixi run dvc metrics show                 # results/metrics/*.json now
pixi run dvc metrics diff                 # vs HEAD (needs the files in git)
pixi run dvc params diff                  # config changes vs HEAD
pixi run dvc plots diff                   # writes dvc_plots/index.html (not committed)
pixi run dvc exp show                     # experiments run from this commit
```

A reproduction matches when `dvc status` is clean after the run, and the
metrics equal the committed ones (exactly for deterministic CPU code;
within a stated tolerance where GPU kernels or threads are
non-deterministic — say which in the write-up). `dvc metrics diff`
compares with `HEAD` only for `cache: false` metrics files committed to
git; a `.gitignore` rule that hides them (check
`git check-ignore -v results/metrics/<file>.json`) leaves `HEAD` empty.

Commit `dvc.lock` (and the `cache: false` metrics) after a run you keep;
`pixi run dvc push` uploads the cached outputs.

## 4. If it does not reproduce

| Symptom | Cause | Fix |
|---|---|---|
| `output '<path>' is already tracked by SCM` | A file inside an output directory is in git (a `.gitkeep`) | `git rm --cached <path>/.gitkeep`; the stage creates the directory |
| `Parameters '<key>' are missing from '<file>'` | `params:` names something that is not a key of the file (a Hydra defaults group such as `model`) | List real keys; depend on the group's YAML file in `deps` |
| `bad DVC file name … is git-ignored` on `dvc add` | A `.gitignore` rule hides the `.dvc` pointer | Add a `!<dir>/*.dvc` exception (`data/raw/` and `data/external/` have one) |
| `ValueError: badly formed help string` from Hydra's argparse | Hydra 1.3 on Python 3.14 (the environment is not the one `pixi.lock` pins, or the `python` pin in `pixi.toml` was lifted) | `pixi install`; keep `python` below 3.14 in `pixi.toml` until Hydra supports it, then `pixi lock` |
| `ConfigAttributeError: Key '<k>' is not in struct`, or a `TypeError` from `instantiate` | The chosen config group option lacks a key the script reads, or its keys do not match its `_target_`'s constructor | Make the option's YAML provide every key the script reads (for `model=`, the class's constructor arguments); `tests/test_configs.py` builds every option |
| Metrics change between identical runs | An unseeded draw, a global RNG used across stages, unordered file listings, threads / GPU non-determinism | The `reproducibility-reviewer` subagent; seed from the config, sort listings |
| Every multirun job reports the same metrics file | Jobs write one fixed path | One `dvc exp run -S …` per point instead |
| A stage reruns every time | A dependency changes on every run (a timestamp, a log, Hydra's `outputs/` inside a dep directory) | Narrow the `deps`, keep run directories out of them |
| A stage never reruns after a code change | The changed file is not in its `deps` | Add the script / package directory / config file |

## 5. Report

Say which commit, which stages ran, whether `dvc status` is clean, and the
metrics before and after (`dvc metrics diff`), with any tolerance and its
source.
