---
name: add-dataset
description: Add a dataset to the research project — raw files tracked with DVC (never git), a reader taken from an upstream library (geostack has no package of its own), a Hydra data config (size, split fractions, seed), the preprocess stage that makes deterministic train / val / test splits, and a test on a tiny fixture. Use when asked to add, ingest, download, load, split or preprocess a dataset, or to add a data config in configs/data or a preprocess script and stage.
---

# Add a dataset

Read "Boundaries" and contracts 1, 3 and 4 in `AGENTS.md` first; this is
the step-by-step.

**Where this repo stands.** geostack has no package of its own (no `src/`),
and until the first dataset lands it has no `configs/` directory, no
preprocess script and no pipeline stage (`dvc.yaml` is `stages: {}`). So the
first dataset creates them, and a reader is not written here: it is taken
from an upstream library. Look before you assume a file exists
(`ls configs scripts`, `cat dvc.yaml`), and say in your report what you had
to create.

## 1. Make sure it does not exist yet

- The capability indexes, one per layer: `docs/api/capabilities.md`
  (GeoStack: the readers of georeader, geotoolz and xrtoolz),
  `capabilities_numerics.md`, `capabilities_geoml.md`,
  `capabilities_geomodels.md`. A reader for the format may exist: use it.
  Say in your report which file you searched and what you found.
- `configs/data/`, if it exists: a new *size* of the same data is a new
  option there, mirroring the keys of the one that exists and chosen with
  `data=<name>`; it is not a new reader, generator or script.
- `pixi run dvc status` and the `*.dvc` pointers: the raw files may already
  be tracked.

## 2. Raw data: DVC, never git

```bash
cp /path/to/source.nc data/raw/            # or download into data/raw/
pixi run dvc add data/raw/source.nc        # writes data/raw/source.nc.dvc
git add data/raw/source.nc.dvc
pixi run dvc push                          # needs a remote: pixi run dvc remote list
```

- `.gitignore` ignores `data/raw/*` but not the `.dvc` pointers
  (`!data/raw/*.dvc`); raw data in another ignored directory needs the
  same exception, or `dvc add` stops with "bad DVC file name … is
  git-ignored" (`git check-ignore -v <pointer>` shows the rule).
- Data you download in a script is a DVC stage with the URL in its `cmd`
  or `dvc import-url <url> data/raw/<file>`, so the download is recorded.
- Keep credentials out of the repo (`.env`, `.dvc/config.local`, both
  gitignored).
- Do not guess what is in a file you have not opened: variable names, the
  name of the time dimension and units are read from the file, or marked
  in the config as to be confirmed.

## 3. The reader comes from upstream

- Use the reader the index lists. The preprocess script calls it with the
  path from the config.
- If no library reads the format, the reader is reusable code and goes
  upstream, into the library that owns the concept, in the lowest layer
  that may hold it, with its tests there (`AGENTS.md`, "Reuse before you
  write"). georeader is not ours and is never changed. Name the library and
  say the dataset waits for that change; do not put the reader in a script,
  a notebook or a new `src/` folder here.
- What is specific to this one dataset and to this one script (which
  variables to keep, how to stack them) may stay in the script as a small
  function that takes plain arguments. The day a second script or a
  notebook needs it, it moves upstream.
- A new third-party library goes into `pixi.toml` (then `pixi lock`);
  xarray, netCDF4, hydra-core and dvc are already there. Check `pixi.toml`
  before importing anything else.

## 4. The config (`configs/data/<name>.yaml`)

A `name`, the size, and the split fractions; choose it with `data=<name>`
or in the `defaults` list of the entry config.

```yaml
name: small
num_samples: 1000
train_split: 0.7
val_split: 0.15
test_split: 0.15
```

The entry config (`configs/<entry>.yaml`, created with the first dataset)
holds the `defaults` list and the seed:

```yaml
defaults:
  - data: small
  - _self_

seed: 0
```

The seed is a config key even when this split needs no random draw (a
split by time), so that every later random step has one source. No path,
fraction, size or seed is a literal in the script.

## 5. Splits and preprocessing (`scripts/preprocess.py`, the `preprocess` stage)

- Split once, here, from the config fractions and the seed
  (`np.random.default_rng(cfg.seed).permutation(n)`), and save the splits
  under `data/processed/`; training and evaluation read the saved splits,
  they never re-split.
- Grouped or temporal data split by group or by time, not by row, so the
  same station or period is never on both sides. Say which you chose and
  why; neighbouring hours of a reanalysis are not independent samples.
- Statistics that transform the data (means, scales, vocabularies,
  imputers) are fitted on the training split only, and saved with it.
- The script is a Hydra entry point that writes under the project root
  (`Path(get_original_cwd()) / "data" / "processed"`), not into Hydra's run
  directory.
- The stage declares everything it reads and writes:

  ```yaml
  stages:
    preprocess:
      cmd: python scripts/preprocess.py
      deps:
        - scripts/preprocess.py
        - data/raw/source.nc
        - configs
      outs:
        - data/processed
  ```

  No git-tracked file may sit inside an output directory (a `.gitkeep`
  makes DVC refuse the stage): the script creates the directory
  (`mkdir(parents=True, exist_ok=True)`), and `.gitignore` ignores all of
  `data/processed/`. `dvc repro` is started through pixi, so the bare
  `python` in `cmd` is the environment's.

## 6. Tests (`tests/test_<name>.py`)

- Write a tiny file into `tmp_path` in the format the reader reads, run the
  preprocessing on it, and check the splits: disjoint, exhaustive, the
  configured sizes, identical for the same seed, and no group or period on
  both sides.
- The missing-path error: `pytest.raises(FileNotFoundError)`.
- No real data and no network in tests. A test that needs Hydra starts
  with `pytest.importorskip("hydra")` and runs under pixi.

## 7. Verify

```bash
pixi run pytest tests/test_<name>.py -v -o addopts=
pixi run dvc dag
pixi run dvc status
pixi run dvc repro preprocess
```

Commit the configs, the script, `dvc.yaml`, `dvc.lock` and the `.dvc`
pointer; never the data. Then the `pre-pr-check` skill.

## 8. Report

Say what you added and what you had to create because the repo did not
have it, which index you searched for a reader and what it holds, what
you ran and what you did not run and why, what is still unconfirmed (names
read from a file you could not open), and what the user has to do next.
