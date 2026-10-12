"""Training entry point using Hydra for configuration management."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from myproject.trainers.trainer import Trainer
from myproject.utils.reproducibility import seed_everything


METRICS_PATH = Path("results") / "metrics" / "train_metrics.json"
LOSS_CURVE_PATH = Path("results") / "figures" / "loss_curve.csv"


def build_model(model_cfg: DictConfig) -> Any:
    """Build the model a ``configs/model/<name>.yaml`` option describes.

    Each option names its class in ``_target_`` and lists its constructor
    arguments; ``name`` labels the option and is not passed on.

    Args:
        model_cfg: The composed ``cfg.model`` node.

    Returns:
        The instantiated model.
    """
    return instantiate({k: v for k, v in model_cfg.items() if k != "name"})


def write_outputs(history: dict[str, list[float]], project_root: Path) -> None:
    """Write the outputs the DVC ``train`` stage declares.

    Per-epoch losses go to the plots CSV (``epoch,train_loss,val_loss``, with
    an empty ``val_loss`` when there is no validation data); the final values
    go to the metrics JSON as a flat dict of numbers.

    Args:
        history: ``Trainer.train``'s per-epoch ``train_loss`` / ``val_loss``.
        project_root: Directory the ``results/`` paths are relative to.
    """
    summary = {
        name: values[-1]
        for name in ("train_loss", "val_loss")
        if (values := history[name])
    }
    metrics_path = project_root / METRICS_PATH
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")

    loss_curve_path = project_root / LOSS_CURVE_PATH
    loss_curve_path.parent.mkdir(parents=True, exist_ok=True)
    val_losses = history["val_loss"]
    with open(loss_curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, train_loss in enumerate(history["train_loss"]):
            val_loss = val_losses[i] if i < len(val_losses) else ""
            writer.writerow([i, train_loss, val_loss])


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object. Automatically populated from
             configs/train.yaml and any overrides passed on the command line.

    Example usage:
        # Use defaults from configs/train.yaml
        python scripts/train.py

        # Override specific parameters
        python scripts/train.py training.lr=0.001 model=transformer

        # Multirun sweep
        python scripts/train.py -m training.lr=0.001,0.01,0.1
    """
    # Set random seed for reproducibility
    seed_everything(cfg.training.seed)

    # Initialize the model the chosen configs/model/<name>.yaml describes
    model = build_model(cfg.model)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
    )

    # Run training (data loading omitted for placeholder)
    history = trainer.train(train_data=None)

    # Write DVC-tracked outputs to project root (Hydra may change CWD)
    project_root = Path(get_original_cwd())
    write_outputs(history, project_root)

    print(f"Training complete. Metrics written to {project_root / METRICS_PATH}")


if __name__ == "__main__":
    main()
