---
title: "Object storage: buckets, keys and five verbs"
subtitle: "Background · explainer · what S3, Azure Blob and GCS can and cannot do, and why formats and catalogs are shaped by it"
short_title: Object storage
description: "A visual explainer of object storage: keys instead of folders, the five operations, range requests, why listing is slow, and how to write several objects safely; background for the storage formats page and the storage design pages."
date: 2026-11-13
keywords:
  - object storage
  - S3
  - Azure Blob Storage
  - range requests
  - obstore
---

+++ {"part": "abstract"}

Background for the storage pages. The running example's B12 band is one object in a bucket; the page follows a window read as HTTP requests, counts what a listing costs, and shows why several writes are not one write. Diagrams carry the explanation.

+++

{button}`Open the full page <./pages/object-storage.html>`

**Status.** Background. The provider facts cited under Sources were checked against the S3, Azure and GCS documentation on 17 November 2026. The obstore `put`, `get_range` and `get_ranges` signatures were checked the same day; the other obstore calls are marked `not pinned`. Nothing on the page is measured.

## TL;DR: a dictionary over HTTP

- **A bucket maps keys to objects.** There are no folders; a `/` in a key is a character.
- **Five verbs:** PUT, GET (whole or a byte range), HEAD, LIST, DELETE. No append, no edit in place, no rename.
- **Range requests** are what make COG, Parquet and Zarr sharding work.
- **Listing is slow** (1 000 keys per page, in sequence), so layouts keep their own index.
- **One PUT is atomic; several are not.** Write new objects, then swap one pointer with a conditional PUT.

## On the page

1. The running example: the B12 band as one object
2. Keys, not folders
3. Five verbs, and the four that are missing
4. Range requests: a window read as HTTP
5. Listing, and when not to
6. Writing safely: one PUT, or a pointer swap
7. S3, Azure and GCS side by side
8. In the stack: geocloud's client pool, geocatalog, pipekit's S3 registry

## Sources and reproduction

Each claim links to the page that supports it:

- S3 is strongly consistent for reads after writes, overwrites, deletes and listings [@aws_s3_consistency].
- Conditional writes (`If-None-Match: *`, `If-Match` with an ETag) fail with 412 Precondition Failed [@aws_s3_conditional_writes; @aws_s3_conditional_requests].
- S3 scales to at least 3 500 writes and 5 500 reads per second per prefix, and can answer 503 Slow Down while it scales [@aws_s3_performance].
- A listing returns up to 1 000 keys per request and continues with a token [@aws_s3_listobjectsv2].
- With a hierarchical namespace, Azure renames or deletes a directory as one atomic operation [@azure_adls_namespace].
- Google Cloud Storage is strongly consistent for reads after writes and for listings [@gcs_consistency], and its `ifGenerationMatch` precondition fails with 412 [@gcs_preconditions].
- The obstore calls in the sketches [@obstore_docs].
