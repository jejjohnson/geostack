---
title: "Recipe registries: versioning code the way data is versioned"
subtitle: "Design · working page · pipekit's content-addressed recipes, and its model registry, cache stores and training data on Icechunk"
short_title: Recipe registries
description: "pipekit already serialises a pipeline to JSON and hashes it; this page stamps every product with the recipe that made it, and puts pipekit's model registry, cache stores and training datasets on Icechunk so code and data share one kind of identity."
date: 2026-11-12
keywords:
  - pipekit
  - provenance
  - model registry
  - Icechunk
  - recipes
---

+++ {"part": "abstract"}

A product is a recipe applied to inputs. [Level 2 collections](./l2_collections.md) versions the inputs and outputs, the [product catalog](./product_catalog.md) records them; this page versions the recipe. It reads pipekit and geotoolz at their 12 November 2026 commits and proposes four workstreams, G to J.

+++

{button}`Open the full page <./pages/recipe-registries.html>`

**Status.** A draft. Workstreams G to J are tracked as GitHub issues.

## TL;DR: a recipe is data

- **Verified today.** `Operator.state` is `{module, class, config}`; `pipeline_hash` is `sha256(stable_json(state) ‖ 0x00 ‖ weights)`; `LocalModelRegistry` and `S3ModelRegistry` store operators under that hash, with tags.
- **G.** Every product records its recipe: the hash, the registry, package versions and a lockfile hash. The hash covers configuration, not code, so versions are kept beside it rather than folded in.
- **H.** An `IcechunkModelRegistry`: a registry store is several writes today; on Icechunk it is one commit.
- **I.** An `IcechunkStore` for `Cache`, in geocloud: carriers stored as Zarr, shared across machines.
- **J.** `SnapshotDataset`: a training dataset whose `content_hash` folds in the snapshot id, so the dataset hash pins the data.

## On the page

1. Current state, verified in code: where pipekit persists things, five gaps
2. Revised target: the same seams on Icechunk
3. G: recipes are versioned like data; config is not code; `RecipeStamp`
4. H, I, J: model registry, cache store, training data
5. Testing gates, sequencing and risks
