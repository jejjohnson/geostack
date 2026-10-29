---
title: toolz primer
subtitle: Report 1 · Where pipekit comes from, and why its operators are classes
short_title: 1 · toolz primer
description: Moved to pipekit's own docs as its Background page. This page points there.
keywords:
  - pipekit
  - toolz
  - functional composition
  - operators
  - carriers
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

This report now lives in pipekit's own documentation, as its [Background](https://jejjohnson.github.io/pipekit/background/toolz/) page. It covers where pipekit comes from, what it inherits from `toolz`, the six things plain functions cannot carry, and why operators settled on a typed base class while the data flowing through them did not. We keep this page only to hold the report's place in the series and to point to the sections.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Moved to [pipekit's docs](https://jejjohnson.github.io/pipekit/background/toolz/); this page is a pointer |
| **Audience** | anyone reviewing the pipekit design before the structural reports |
| **Companions** | [Report 2](02_pipekit_surface.md) (pipekit's shipped surface), [Report 3](03_sister_libraries.md) (sister libraries), [Report 9](09_use_cases.md) (use cases) |
| **Checked against** | toolz 1.2.0 and pipekit 0.0.2 (`ef3c17f`), when it moved |
:::

(r1-moved)=
## Where each section went

We moved the report because a design that lives with the code is read by the people who use the code, and it stays checked against it. It moved whole, with its examples, tables and figures, in MkDocs syntax ([](#tbl-r1-moved)).

:::{table} The report's sections and their new homes on pipekit's Background page.
:label: tbl-r1-moved

| Section here | On the Background page |
|---|---|
| Why this report exists | [Why this page exists](https://jejjohnson.github.io/pipekit/background/toolz/#why-this-page-exists) |
| toolz as the ancestor, and the one-to-one mapping to pipekit | [toolz as the ancestor](https://jejjohnson.github.io/pipekit/background/toolz/#toolz-as-the-ancestor) |
| Where toolz stops being enough (the six gaps) | [Where toolz stops being enough](https://jejjohnson.github.io/pipekit/background/toolz/#where-toolz-stops-being-enough) |
| The trajectory: functions, Protocols, typed entities | [The trajectory](https://jejjohnson.github.io/pipekit/background/toolz/#the-trajectory-functions-protocols-typed-entities) |
| Why typed operators are not over-engineering | [Why typed operators are not over-engineering](https://jejjohnson.github.io/pipekit/background/toolz/#why-typed-operators-are-not-over-engineering) |
| Where next | [Where next](https://jejjohnson.github.io/pipekit/background/toolz/#where-next) |
:::

One correction travelled with it. Nested pipelines can be *loaded* from hand-written Hydra YAML today (`HydraConfigLoader.from_yaml` in pipekit-experiment), whereas what is missing is dumping a pipeline built in Python.

We cover the shipped surface in [Report 2](02_pipekit_surface.md), which now points to pipekit's [tour of the core](https://jejjohnson.github.io/pipekit/core-tour/).
