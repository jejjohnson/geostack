"""Training entry point using hydra-zen for type-safe config management.

hydra-zen benefits over pure Hydra:
- No YAML boilerplate: configs are defined as Python dataclasses
- Type-safe: configs are validated at runtime using Python type hints
- Auto-generated: configs are automatically generated from function signatures
- Composable: configs can be composed programmatically

It configures the same run as ``scripts/train.py``: the same models (one
``model`` group option per ``configs/model/<name>.yaml``, with the same
defaults), learning rate, epochs and seed. Override them on the command line::

    python scripts/train_zen.py model=transformer trainer.lr=0.01 seed=0
"""

from __future__ import annotations

from typing import Any

from hydra_zen import ZenStore, builds, make_config, zen

from geostack.models.baseline import BaselineModel
from geostack.models.transformer import TransformerModel
from geostack.trainers.trainer import Trainer
from geostack.utils.reproducibility import seed_everything


# --- Build structured configs from Python classes ---
# builds() generates a dataclass config from a class's signature, so the
# defaults cannot drift from the class (they match configs/model/*.yaml).
MODEL_CONFIGS = {
    "baseline": builds(BaselineModel, populate_full_signature=True),
    "transformer": builds(TransformerModel, populate_full_signature=True),
}

# A partial: the model is passed in when the trainer is built. The defaults
# match the `training` keys of configs/train.yaml.
TrainerConfig = builds(
    Trainer,
    lr=1e-3,
    epochs=100,
    populate_full_signature=True,
    zen_partial=True,
)

# --- Compose a top-level config ---
# The `model` group defaults to the baseline, as in configs/train.yaml.
ExperimentConfig = make_config(
    hydra_defaults=["_self_", {"model": "baseline"}],
    model=None,
    trainer=TrainerConfig,
    seed=42,
)

# --- Register configs in the Zen store ---
store = ZenStore()
for name, config in MODEL_CONFIGS.items():
    store(config, name=name, group="model")
store(ExperimentConfig, name="experiment")


def run_experiment(model: Any, trainer: Any) -> dict:
    """Run the experiment with the instantiated model and trainer factory.

    Args:
        model: The model the chosen ``model`` option builds.
        trainer: A partial ``Trainer`` that takes the model.

    Returns:
        The per-epoch training history.
    """
    metrics = trainer(model=model).train(train_data=None)
    print(f"Training {model!r} complete. Metrics: {metrics}")
    return metrics


def seed_before_instantiation(cfg: Any) -> None:
    """Seed every RNG before zen instantiates the model from the config.

    Args:
        cfg: The composed experiment config.
    """
    seed_everything(cfg.seed)


if __name__ == "__main__":
    store.add_to_hydra_store()
    # zen(...).hydra_main is hydra-zen's equivalent of @hydra.main: it parses
    # command-line overrides, composes the config, instantiates `model` and
    # `trainer` and passes them to run_experiment.
    zen(run_experiment, pre_call=seed_before_instantiation).hydra_main(
        config_name="experiment",
        config_path=None,
        version_base="1.3",
    )
