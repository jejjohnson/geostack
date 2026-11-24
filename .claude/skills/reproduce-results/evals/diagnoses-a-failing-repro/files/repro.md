I ran `pixi run dvc repro` and got three problems. This is the stage:

```yaml
stages:
  train:
    cmd: python scripts/train.py model=mlp
    deps:
      - scripts/train.py
      - outputs
    params:
      - configs/train.yaml:
          - training
          - model
    outs:
      - results/checkpoints
    metrics:
      - results/metrics/train_metrics.json:
          cache: false
```

1. First run: `ERROR: failed to reproduce 'train': Parameters 'model' are missing from 'configs/train.yaml'.`
2. After I removed `model` from params it ran, but now it reruns every single time even when nothing changed.
3. When I edit the model class in the upstream library and run `dvc repro`, it says everything is up to date.
4. `results/metrics/train_metrics.json` differs in the fourth decimal between two identical runs.

`scripts/train.py` uses Hydra and writes its run directory to `outputs/`. `git ls-files results/checkpoints` prints `results/checkpoints/.gitkeep`.
