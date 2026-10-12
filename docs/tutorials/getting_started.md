# Getting Started

## Prerequisites

Install [Pixi](https://pixi.sh) for environment management:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Installation

Clone the repository and install the environment:

```bash
git clone https://github.com/jejjohnson/geostack
cd geostack
pixi install
```

## Running Experiments

geostack has no experiments yet. Each one will be a script in `scripts/` that
composes the upstream libraries, a Hydra config in `configs/` and a stage in
`dvc.yaml` (the `add-experiment` skill), run with:

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
