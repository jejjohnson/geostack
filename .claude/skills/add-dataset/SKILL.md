---
name: add-dataset
description: Add a dataset to the research project — raw files tracked with DVC (never git), a loader in the package's data subpackage, a Hydra data config (size, split fractions), the preprocess stage that makes deterministic train / val / test splits, and tests on a tiny fixture. Use when asked to add, ingest, download, load, split or preprocess a dataset, or to add a data loader or data config in src/myproject/data, configs/data or scripts/preprocess.py.
---

# Add a dataset

<!-- template-example: example-specific paths (src/myproject/data, configs/data, scripts/preprocess.py); keep the procedure, rewrite the names after adoption. -->

Read "Boundaries" and contracts 1, 3 and 4 in `AGENTS.md` first; this is
the step-by-step.

## 1. Make sure it does not exist yet

- `docs/api/capabilities.md` (`myproject.data.loading`): a loader for the
  format may exist — extend it rather than writing a second one.
- `configs/data/` (`small.yaml`, `full.yaml`): a new *size* of the same data
  is a new option there, not a new loader.
- `pixi run dvc status` and the `*.dvc` pointers: the raw files may already
  be tracked.

## 2. Raw data: DVC, never git

```bash
cp /path/to/source.nc data/raw/            # or download into data/raw/
pixi run dvc add data/raw/source.nc        # writes data/raw/source.nc.dvc
git add data/raw/source.nc.dvc
pixi run dvc push                          # needs a remote: dvc remote add -d <name> <url>
```

- `.gitignore` ignores `data/raw/*` but not the `.dvc` pointers
  (`!data/raw/*.dvc`); raw data in another ignored directory needs the
  same exception, or `dvc add` stops with "bad DVC file name … is
  git-ignored" (`git check-ignore -v <pointer>` shows the rule).
- Data you download in a script is a DVC stage with the URL in its `cmd`
  or `dvc import-url <url> data/raw/<file>`, so the download is recorded.
- Keep credentials out of the repo (`.env`, `.dvc/config.local`, both
  gitignored).

## 3. The loader (`src/myproject/data/`)

- A function in `loading.py` (or a new module for a new family of
  formats) that takes the path as `str | Path` and returns arrays /
  `xarray` objects / a dict of them, raising `FileNotFoundError` with the
  path when it is missing — as `load_raw_data` does.
- No config, no global state, no hard-coded path, no randomness: the
  script passes the path and the seed.
- Type hints and a Google-style docstring saying what is returned (shapes,
  units, dims).
- A new third-party reader (netCDF4, xarray, pandas, …) goes into
  `pixi.toml` (then `pixi lock`); xarray and netCDF4 are already there.

## 4. The config (`configs/data/<name>.yaml`)

Mirror `configs/data/small.yaml`: a `name`, the size, and the split
fractions; choose it with `data=<name>` (or as the default in
`configs/train.yaml`'s `defaults` list).

```yaml
name: small
num_samples: 1000
train_split: 0.7
val_split: 0.15
test_split: 0.15
```

## 5. Splits and preprocessing (`scripts/preprocess.py`, the `preprocess` stage)

- Split once, here, from the config fractions and the seed
  (`np.random.default_rng(cfg.training.seed).permutation(n)`), and save
  the splits under `data/processed/`; training and evaluation read the
  saved splits, they never re-split.
- Statistics that transform the data (means, scales, vocabularies,
  imputers) are fitted on the training split only, and saved with it.
- Grouped or temporal data split by group / time, not by row, so the same
  site or period is never on both sides.
- The `preprocess` stage in `dvc.yaml` lists the raw files, the script,
  `src/myproject/data` and the data config as `deps`, and `data/processed`
  as its `outs`. No git-tracked file may sit inside an output directory
  (a `.gitkeep` makes DVC refuse the stage): the script creates the
  directory (`mkdir(parents=True, exist_ok=True)`), and `.gitignore`
  ignores all of `data/processed/`.

## 6. Tests (`tests/test_data.py`)

- Write a tiny file into `tmp_path` in the format the loader reads, load
  it, and check values, shapes and dtypes — as
  `test_load_raw_data_returns_dict` does with an empty directory.
- The missing-path error: `pytest.raises(FileNotFoundError)`.
- A split helper in the package: disjoint, exhaustive, the configured
  sizes, identical for the same seed.
- No real data and no network in tests.

## 7. Export and index

`make capabilities` after adding a public function, and commit
`docs/api/capabilities.md`.

## 8. Verify

```bash
uv run pytest tests/test_data.py -v -o addopts=
pixi run dvc status
pixi run dvc repro preprocess
make capabilities && uv run pytest tests/test_capabilities.py -o addopts=
```

Then the `pre-pr-check` skill.
