---
title: "Processing levels: the operators between L0 and L5"
subtitle: "Background · tutorial · what each transition does to the data, what kind of operator performs it, and what software it needs underneath"
short_title: Processing levels
description: "A visual tutorial on the L0 to L5 data lifecycle: the five transitions written as operators, the eight engineering jobs under them and the carrier switch between L2 and L3, with sea surface temperature simulated from detector counts to heatwave events and scored against a known truth."
date: 2026-11-14
keywords:
  - processing levels
  - calibration
  - retrieval
  - optimal interpolation
  - data assimilation
  - marine heatwaves
---

+++ {"part": "abstract"}

Background for axis 2 of the [overview](../overview.md), the L0 to L5 lifecycle and its engineering rows. The page is about the arrows between the levels. One variable, sea surface temperature, is simulated from detector counts to heatwave events, and every level is scored against the truth. The example uses no GeoStack package, and each transition lists the tools the wider community uses for it.

+++

{button}`Open the full page <./pages/processing-levels.html>`

**Status.** Background. The example is one file of NumPy and SciPy, run in November 2026 on a synthetic ocean, atmosphere and sensor; the model in the L4 step knows the true dynamics, so its numbers illustrate the transitions and say nothing about real products. Agency level definitions and the named operational systems are given from memory and were not re-checked online.

## TL;DR: five transitions, five kinds of operator

- **Each transition changes one thing:** units (L0→L1), meaning (L1→L2), geometry and completeness (L2→L3), time (L3→L4), the question answered (L4→L5).
- **Each is a different kind of operator:** a lookup per detector, an inversion per pixel, one global solve per map, a cycle that needs yesterday's result, a reduction over years that ends in a table.
- **The data change shape once,** between L2 (a swath: every pixel has its own position) and L3 (a cube on a shared grid).
- **Accuracy and completeness trade:** L2 is right to 0.19 K but covers 31 % of the map on a given day; L3 covers everything at 0.47 K; L4, with dynamics, 0.27 K.
- **Every level carries its uncertainty forward:** the L3 variance is the L4 observation error.
- **The test:** the heatwave found from L4 matches 68 % of the true heatwave cell-days; from L3 alone, 34 %.

## On the page

1. The six levels, and how their names differ from the NASA and GHRSST conventions
2. The running example, with one day shown at every level
3. Part 1: every transition as an operator (inputs, outside information, fitted parameters, constants, state, outputs), then each in turn with its code, its numbers, and what the software has to solve
4. The numbers level by level, and a units check
5. Five kinds of operator, side by side
6. Part 2: the eight engineering jobs, on both sides of the carrier switch
7. Six design ideas that recur

## Sources and reproduction

Every measured number on the page comes from `notebooks/background/processing_levels.py` (run by hand: `pixi run -e background python notebooks/background/processing_levels.py`; its first cell lists what it needs). The recorded output is `docs/images/background_processing_levels/` (`results.json` and the six panels), and the page was built from it. It is seeded and reads no data, so a rerun gives the same numbers; it is not a DVC stage yet.

Sources: the level names follow the [overview](../overview.md); the marine heatwave definition is that of Hobday et al. [@hobday2016].
