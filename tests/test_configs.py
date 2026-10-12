"""Every ``model=`` option composes and builds, the same way in both entry points.

``scripts/train.py`` builds ``cfg.model`` from its ``_target_`` whichever
``configs/model/<name>.yaml`` is chosen, and ``scripts/train_zen.py`` offers
the same options with the same defaults. Hydra and hydra-zen live in the pixi
environment only, so these tests skip elsewhere (``make test`` in the uv
environment).
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
MODEL_OPTIONS = sorted(p.stem for p in (CONFIGS / "model").glob("*.yaml"))


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name="train", overrides=overrides)


@pytest.mark.parametrize("option", MODEL_OPTIONS)
def test_model_option_builds_with_train_script(option: str) -> None:
    train = _load_script("train")
    cfg = _compose([f"model={option}"])
    for key in ("lr", "epochs", "seed"):
        assert key in cfg.training
    model = train.build_model(cfg.model)
    assert type(model).__name__ in repr(model)


def test_train_zen_offers_the_same_models_with_the_same_defaults() -> None:
    pytest.importorskip("hydra_zen")
    from hydra_zen import instantiate

    train = _load_script("train")
    train_zen = _load_script("train_zen")
    assert sorted(train_zen.MODEL_CONFIGS) == MODEL_OPTIONS
    for option, config in train_zen.MODEL_CONFIGS.items():
        from_yaml = train.build_model(_compose([f"model={option}"]).model)
        assert repr(instantiate(config)) == repr(from_yaml)

    trainer = instantiate(train_zen.TrainerConfig)(model=None)
    training = _compose([]).training
    assert (trainer.lr, trainer.epochs) == (training.lr, training.epochs)
    assert train_zen.ExperimentConfig.seed == training.seed


def test_train_outputs_have_the_shapes_the_notebook_reads(tmp_path: Path) -> None:
    """Scalars in the metrics JSON, one row per epoch in the plots CSV."""
    train = _load_script("train")
    history = {"train_loss": [1.0, 0.5, 0.25], "val_loss": [1.5, 0.75, 0.5]}
    train.write_outputs(history, tmp_path)

    metrics = json.loads((tmp_path / train.METRICS_PATH).read_text())
    assert metrics == {"train_loss": 0.25, "val_loss": 0.5}

    with open(tmp_path / train.LOSS_CURVE_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [float(row["train_loss"]) for row in rows] == history["train_loss"]
    assert [float(row["val_loss"]) for row in rows] == history["val_loss"]
    assert [int(row["epoch"]) for row in rows] == [0, 1, 2]


def test_train_outputs_without_validation_data(tmp_path: Path) -> None:
    train = _load_script("train")
    train.write_outputs({"train_loss": [1.0, 0.5], "val_loss": []}, tmp_path)
    metrics = json.loads((tmp_path / train.METRICS_PATH).read_text())
    assert metrics == {"train_loss": 0.5}
    lines = (tmp_path / train.LOSS_CURVE_PATH).read_text().splitlines()
    assert lines == ["epoch,train_loss,val_loss", "0,1.0,", "1,0.5,"]
