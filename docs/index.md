# GeoStack

Welcome to the documentation for GeoStack.

## Overview

A vertically integrated, JAX-native research software stack for geophysical
modeling, inference and data assimilation. Start with the
[GeoStack overview](overview.md): the problem, the three organising axes, the
package map and an end-to-end walkthrough.

geostack has no package of its own: scripts and notebooks here compose the
libraries it builds on,
[pipekit](https://github.com/jejjohnson/pipekit),
[geotoolz](https://github.com/jejjohnson/geotoolz),
[xrtoolz](https://github.com/jejjohnson/xrtoolz) and
[georeader](https://github.com/spaceml-org/georeader), pinned as git
submodules under `references/`. Reusable code goes into those libraries.

## Features

- **Pixi** environment management with conda-forge packages
- **DVC** for data versioning and pipeline management
- **Hydra** + **hydra-zen** for configuration management
- **MyST** for scientific documentation
- **JupyterLab** and **Marimo** notebook environments
- **Agent-ready**: `AGENTS.md` and `CODE_REVIEW.md` for coding agents, and a
  [capability index](api/capabilities.md) of the libraries geostack builds on

## Getting Started

See the [Getting Started tutorial](tutorials/getting_started.md) to set up
your environment.
