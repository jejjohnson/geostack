# API Reference

<!-- template-example: the modules, signatures and summaries below are the example package's; rewrite them from your own package after adopting the template. -->

The public API of the `myproject` package, by module. The
[capability index](capabilities.md) lists the same names with the first
sentence of each docstring and is regenerated from the code
(`make capabilities`); this page adds the signatures. The docstrings in
`src/myproject/` are the full reference.

## `myproject.data.loading`

Placeholder data loading functions.

```python
load_raw_data(data_dir: str | Path) -> dict
```

Load raw data from `data_dir`. Returns `{"data": ..., "labels": ...}`;
raises `FileNotFoundError` if the directory does not exist.

```python
load_processed_data(data_dir: str | Path) -> dict
```

Load processed data from `data_dir`. Returns the splits
`{"train": ..., "val": ..., "test": ...}`; raises `FileNotFoundError` if the
directory does not exist.

## `myproject.models.baseline`

```python
class BaselineModel(hidden_size: int = 64, num_layers: int = 2)
```

A simple baseline model; `forward(x)` runs the forward pass. Configured by
`configs/model/baseline.yaml` (`model=baseline`, the default).

## `myproject.models.transformer`

```python
class TransformerModel(d_model: int = 256, nhead: int = 8, num_layers: int = 6)
```

A placeholder Transformer encoder; `forward(x)` runs the forward pass.
Configured by `configs/model/transformer.yaml` (`model=transformer`).

## `myproject.trainers.trainer`

```python
class Trainer(model: Any, lr: float = 1e-3, epochs: int = 10)
```

Runs experiments. `Trainer.train(train_data, val_data=None) -> dict` returns
the per-epoch history `{"train_loss": [...], "val_loss": [...]}`
(`val_loss` is empty without validation data).

## `myproject.utils.reproducibility`

```python
seed_everything(seed: int = 42) -> None
```

Seeds Python's `random`, NumPy and, when it is installed, PyTorch (with
deterministic cuDNN). Entry points call it once, with the seed from the
config.
