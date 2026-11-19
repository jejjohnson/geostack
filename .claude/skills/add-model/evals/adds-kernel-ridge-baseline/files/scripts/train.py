"""Fit the configured model and write its metrics."""

import json
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf


def build_model(model_cfg: DictConfig):
    """Instantiate the option's _target_ with every key but `name`."""
    kwargs = {k: v for k, v in OmegaConf.to_container(model_cfg).items() if k != "name"}
    return instantiate(kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    train = np.load(root / "data" / "processed" / "train.npy")
    test = np.load(root / "data" / "processed" / "test.npy")
    model = build_model(cfg.model)
    model.fit(train[:, 1:], train[:, 0])
    prior = cfg.retrieval.wind_speed_prior
    residual = model.predict(test[:, 1:]) - test[:, 0]
    penalty = cfg.retrieval.regularisation * prior
    metrics = {"rmse": float(np.sqrt(np.mean(residual**2))), "penalty": float(penalty)}
    out = root / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_metrics.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
