---
title: "Parallelism: threads, processes and async"
subtitle: "Background · explainer · why reading 121 tiles takes 6.4 s one at a time, and which mechanism gets it to 0.23 s"
short_title: Parallelism
description: "A visual explainer of concurrency and parallelism in Python: waiting versus computing, async, threads and the GIL, processes, vectorising and distributing, with measured timings for one Sentinel-2 band; background for the storage formats page and the storage design pages."
date: 2026-11-13
keywords:
  - concurrency
  - parallelism
  - asyncio
  - threads
  - GIL
  - multiprocessing
---

+++ {"part": "abstract"}

Background for the storage pages that follow it. One running example, reading all 121 tiles of band B12 of a Sentinel-2 L2A tile, goes through every mechanism: serial, async, threads and processes, each drawn as a timeline and timed. Diagrams carry the explanation; the maths sits in folded panels.

+++

{button}`Open the full page <./pages/parallelism.html>`

**Status.** Background. Timings measured in November 2026: 121 tiles, 50 ms injected latency per fetch, real deflate decode, CPython 3.13 with the GIL, 2 cores. The 500-scene figures are extrapolated, not measured.

## TL;DR: waiting and computing need different tools

- **Two kinds of work.** Waiting on a network or disk leaves the CPU free; computing keeps it busy. A cloud read is mostly waiting: 50 ms of fetch to 2.3 ms of decode per tile.
- **Async** overlaps waits on one thread and never computes two things at once: 0.38 s with 64 requests in flight, and 6.39 s if one blocking call hides inside it.
- **Threads** overlap waits, and computing too when it runs in NumPy, zlib or Rust, which release the GIL: 0.23 s with 64 threads, against 6.40 s serial.
- **Processes** overlap any computing, including pure Python, and pay in start-up time and copying; for this read they are expensive threads (3.25 s with 2).
- **Below and above.** Vectorising speeds up one worker (47× for scaling a tile); distributing adds machines.

## On the page

1. The running example: 121 tiles, one wait and one computation each
2. Two kinds of work, and concurrency against parallelism
3. Part 1: async, threads (and the GIL), processes, each with a timeline, a code sketch, the measured time and when to use it
4. Part 2: vectorising below, distributing above
5. Choosing: a decision guide and a comparison table
6. The measured numbers with the model behind them
7. In the stack: where geotoolz, pipekit, zarr, dask and JAX use each mechanism, and calling async code from blocking code
