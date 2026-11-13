---
title: "Versions and identity: hashes, snapshots, commits"
subtitle: "Background · explainer · how to say which data, code and settings made a product, and why git, DVC, Icechunk and pipekit share one idea"
short_title: Versions and identity
description: "A visual explainer of content hashes, immutable snapshots, branches and tags, atomic commits and provenance, measured on an Icechunk repository; background for the Level 2 collections and recipe registries designs."
date: 2026-11-13
keywords:
  - versioning
  - content hash
  - snapshot
  - Icechunk
  - DVC
  - provenance
---

+++ {"part": "abstract"}

Background for the Level 2 collections and recipe registries pages. The running example's B12 band is ingested into an Icechunk repository, then reprocessed over the site; the page follows what is stored, what each version reads, and what a commit guarantees. Diagrams carry the explanation.

+++

{button}`Open the full page <./pages/versions-identity.html>`

**Status.** Background. Measured in November 2026 with icechunk 2.3.1 and zarr 3.4.2 on a synthetic band in a local repository.

## TL;DR: never overwrite; name every state

- **A hash** is a fingerprint of bytes: one pixel changed, 63 of 64 digits differ.
- **A snapshot** is a list of pointers to immutable pieces; a new version stores only what changed (4 of 121 chunks).
- **A branch** is a name that moves; a **tag** is a name that does not; an id is forever.
- **A commit** is one conditional pointer move: all or nothing, and the second of two writers gets a conflict.
- **Provenance** is the ids of data, code, environment and settings recorded on the product.
- **DVC and Icechunk fit together:** DVC keeps the pipeline and ordinary files, Icechunk the arrays, and a pointer file holding the snapshot id joins them under one git commit.

## On the page

1. The running example: ingest, then reprocess the site window
2. Identity: content hashes
3. Snapshots that share what did not change
4. History: branches and tags
5. Commits: all or nothing
6. Provenance: ids on the product
7. One idea in git, DVC, Icechunk, pipekit, lock files and images
8. DVC and Icechunk together: a pointer file joins them, run end to end
9. In the stack, and where each idea goes in the design pages
