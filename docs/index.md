# GeoStack

Welcome to the documentation for GeoStack.

## Overview

A vertically integrated, JAX-native research software stack for geophysical
modeling, inference and data assimilation. Start with the
[GeoStack overview](overview.md): the problem, the three organising axes, the
package map and an end-to-end walkthrough. Its
[reading list](overview.md#sec-reading) links every report, the GeoModels
pages and the benchmarks gallery.

geostack has no package of its own: scripts and notebooks here compose the
libraries it builds on,
[pipekit](https://github.com/jejjohnson/pipekit),
[geotoolz](https://github.com/jejjohnson/geotoolz),
[xrtoolz](https://github.com/jejjohnson/xrtoolz) and
[georeader](https://github.com/spaceml-org/georeader), pinned as git
submodules under `references/`. Reusable code goes into those libraries.

::::{grid} 1 1 3 3

:::{card} Overview
:link: overview.md
The map of the stack: three organising axes, the tiers, a methane
walkthrough, open questions and the plan.
:::

:::{card} Examples
:link: overview_examples.md
Code sketches for every pattern in the overview, from operators to
reproducibility artifacts.
:::

:::{card} Capability index
:link: api/capabilities.md
Every public name in the libraries GeoStack builds on. Search it before
writing new code.
:::
::::

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
