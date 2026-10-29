# Getting Started

## Prerequisites

Install [Pixi](https://pixi.sh) for environment management:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Installation

Clone the repository and install the environment:

```bash
git clone --recurse-submodules https://github.com/jejjohnson/geostack
cd geostack
# already cloned? git submodule update --init
pixi install
```

## Running Experiments

geostack has no experiments yet. Each one will be 1) a script in `scripts/` which composes the upstream libraries, 2) a Hydra config in `configs/` and 3) a stage in `dvc.yaml` (see the `add-experiment` skill). We run them with:

```bash
pixi run dvc repro
```

## Notebook Environments

### JupyterLab

```bash
pixi run -e jupyterlab lab
```

### Marimo

```bash
pixi run -e marimo marimo-edit
```
