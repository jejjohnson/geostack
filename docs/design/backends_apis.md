---
title: "Backends and APIs: REST, FastAPI and LitServe"
subtitle: "Background · explainer · how a request from outside becomes a product, and the difference between the STAC we query and the catalog we serve"
short_title: Backends and APIs
description: "A visual explainer of HTTP and REST, FastAPI, long-running jobs, the blocking trap and LitServe, with a small measured service over the NDVI pipeline; it separates upstream STAC APIs, our product database and our product catalog."
date: 2026-11-13
keywords:
  - REST
  - FastAPI
  - LitServe
  - STAC
  - product catalog
  - jobs
---

+++ {"part": "abstract"}

Background for the [product catalog](./product_catalog.md) design and for services built on the stack. It wraps the NDVI pipeline of [the geo pipeline](./geo_pipeline.md) in a small FastAPI service, runs it, and measures it. One distinction runs through the page: the catalogs we query are not the catalog we serve.

+++

{button}`Open the full page <./pages/backends-apis.html>`

**Status.** Background. The service is one file, run in November 2026 with FastAPI 0.162 over geotoolz at `640670c`; SQLite stands in for the product database, the job worker is a thread in the web process, and the 200 ms read is an injected delay. LitServe is described from its repository page and was not run. The product database and product catalog are proposed (geotoolz#671 to #674).

## TL;DR: a thin, typed, forgetful layer over the pipeline

- **REST** is a convention: resources with addresses, a few verbs, JSON answers with a status code, no memory between requests.
- **Two STACs.** Upstream, we are a client of a provider's STAC API. Downstream, the pipeline writes our product database, which is mirrored into a product catalog that we serve as STAC.
- **FastAPI** turns a typed function into an endpoint: it parses, validates, serialises and documents from the type hints.
- **Long work is a job:** accept, return an id at once (4 ms here), let the client check back; the id is a hash of the request, so a retry cannot run it twice.
- **The blocking trap:** one blocking call in an async endpoint turned 20 concurrent requests from 0.22 s into 4.03 s and held an unrelated health check for 4.0 s.
- **LitServe** is FastAPI plus what model serving needs (load once, batch, spread over GPUs); it belongs at the predict endpoint.

## On the page

1. The running example: the NDVI pipeline behind an API
2. HTTP and REST: one exchange, the verbs, the status codes
3. Two STACs: upstream, the product database, the product catalog
4. FastAPI: a typed function becomes an endpoint
5. Four kinds of endpoint: search, read, run, predict
6. Long work is a job, with the idempotency key
7. The blocking trap, measured
8. Stateless services
9. LitServe: FastAPI for models
10. In the stack, and what is left for later
