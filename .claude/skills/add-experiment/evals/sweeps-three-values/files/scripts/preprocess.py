"""Make the synthetic CH4 samples and their train / val / test splits."""

from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    rng = np.random.default_rng(cfg.training.seed)
    n = cfg.data.num_samples
    columns = 1900.0 + 20.0 * rng.standard_normal((n, 8))
    order = rng.permutation(n)
    n_train = int(cfg.data.train_split * n)
    n_val = int(cfg.data.val_split * n)
    out = Path(get_original_cwd()) / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train.npy", columns[order[:n_train]])
    np.save(out / "val.npy", columns[order[n_train : n_train + n_val]])
    np.save(out / "test.npy", columns[order[n_train + n_val :]])


if __name__ == "__main__":
    main()
