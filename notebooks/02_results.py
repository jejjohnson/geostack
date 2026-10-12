# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Results Analysis
#
# This notebook analyzes and visualizes experiment results.
#
# > **Note**: This file is a [Jupytext](https://jupytext.readthedocs.io) percent-format
# > script. To open it as a Jupyter notebook, run:
# > ```bash
# > pixi run notebooks-to-ipynb
# > ```
# > or convert a single file with:
# > ```bash
# > pixi run jupytext --to notebook notebooks/02_results.py
# > ```

# %% [markdown]
# ## Imports

# %%
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

# %% [markdown]
# ## Load Results
#
# The DVC `train` stage writes the final losses to
# `results/metrics/train_metrics.json` (scalars, compared by `dvc metrics diff`)
# and the per-epoch losses to `results/figures/loss_curve.csv` (its plot).
# Run `pixi run dvc repro` first; from the `notebooks/` directory the paths
# are one level up.

# %%
root = Path.cwd() if Path("results").exists() else Path.cwd().parent
metrics_path = root / "results" / "metrics" / "train_metrics.json"
loss_curve_path = root / "results" / "figures" / "loss_curve.csv"

if metrics_path.exists() and loss_curve_path.exists():
    metrics = json.loads(metrics_path.read_text())
    with open(loss_curve_path, newline="") as f:
        rows = list(csv.DictReader(f))
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows if row["val_loss"]]
    print("Loaded metrics:", metrics)
else:
    # Placeholder results when no experiment has been run yet
    train_loss = [1.0, 0.8, 0.6, 0.4, 0.3]
    val_loss = [1.1, 0.9, 0.7, 0.5, 0.4]
    metrics = {"train_loss": train_loss[-1], "val_loss": val_loss[-1]}
    print("No results found — using placeholder data.")

# %% [markdown]
# ## Training Curves

# %%
fig, ax = plt.subplots(figsize=(8, 4))

if train_loss:
    ax.plot(train_loss, label="Train loss", marker="o")
if val_loss:
    ax.plot(val_loss, label="Val loss", marker="s", linestyle="--")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Curves")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final Metrics Summary

# %%
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
