# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Background · backends and APIs: the service
#
# The FastAPI service shown on `docs/design/pages/backends-apis.html`. It is started
# and exercised by `backends.py`; it is not meant to be run on its own.

# %%
"""A small geo backend: search our products, run a recipe as a job, three ways to read.

FastAPI in front of the pipeline of the geo-pipeline page. SQLite stands in for
the product database (PostGIS in production); Icechunk holds the products.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import datetime
import glob
import hashlib
import json
import os
import pathlib
import sqlite3
import time
from typing import Literal

import fastapi
import geocatalog as gc
import geoops as gz
import geopatcher as gp
import icechunk
import numpy as np
import pandas as pd
import pipekit_ops
import pydantic
import pyproj
import zarr
from geocatalog.build import build_raster_catalog
from geoops.patch_ops import ApplyToChips, GridSampler, MergePatches

HERE = pathlib.Path(__file__).parent
ARCHIVE = pathlib.Path(os.environ["BACKGROUND_ARCHIVE"])  # set by backends.py
STATE = pathlib.Path(os.environ["BACKGROUND_STATE"])
CRS = pyproj.CRS.from_epsg(32630)


# === models ===
class RunRequest(pydantic.BaseModel):
    """What a client sends to start a job: a site, a period and a recipe."""

    bounds: tuple[float, float, float, float]
    start: datetime.date
    end: datetime.date
    recipe: Literal["ndvi"] = "ndvi"


class Product(pydantic.BaseModel):
    """One row of our product database, as the API returns it."""

    id: str
    start_time: datetime.date
    end_time: datetime.date
    bbox: tuple[float, float, float, float]
    href: str
    snapshot_id: str
    recipe_hash: str


class Job(pydantic.BaseModel):
    """A run in progress or finished; product_id is set once it is done."""

    id: str
    status: Literal["queued", "running", "done", "failed"]
    product_id: str | None = None


# === database ===
def database() -> sqlite3.Connection:
    connection = sqlite3.connect(STATE / "products.sqlite")
    connection.row_factory = sqlite3.Row
    return connection


def create_tables() -> None:
    with database() as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS products (id TEXT PRIMARY KEY, start_time TEXT,"
            " end_time TEXT, x0 REAL, y0 REAL, x1 REAL, y1 REAL, href TEXT,"
            " snapshot_id TEXT, recipe_hash TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT,"
            " product_id TEXT)"
        )


def product_from_row(row: sqlite3.Row) -> Product:
    bbox = (row["x0"], row["y0"], row["x1"], row["y1"])
    return Product(bbox=bbox, **{key: row[key] for key in Product.model_fields if key != "bbox"})


def job_from_row(row: sqlite3.Row) -> Job:
    return Job(id=row["id"], status=row["status"], product_id=row["product_id"])


# === recipe ===
NDVI = gz.DNToReflectance(scale=1e-4) | gz.NDVI(red=0, nir=1)
RECIPE_HASH = pipekit_ops.pipeline_hash(NDVI, b"")


def run_recipe(job_id: str, request: RunRequest, catalog: gc.GeoCatalog) -> None:
    """Run the job and record how it ended: a job never stays 'running'."""
    with database() as db:
        db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job_id,))
    try:
        product_id = run_stages(job_id, request, catalog)
    except Exception:
        # nobody holds the future, so the job row is the only record of a failure
        with database() as db:
            db.execute("UPDATE jobs SET status = 'failed' WHERE id = ?", (job_id,))
        raise
    with database() as db:
        db.execute(
            "UPDATE jobs SET status = 'done', product_id = ? WHERE id = ?",
            (product_id, job_id),
        )


def run_stages(job_id: str, request: RunRequest, catalog: gc.GeoCatalog) -> str:
    """The ten stages of the pipeline page, ending in one row of the product table."""
    interval = pd.Interval(
        pd.Timestamp(request.start), pd.Timestamp(request.end), closed="both"
    )
    site = gc.GeoSlice(
        bounds=request.bounds, interval=interval, resolution=(10.0, 10.0), crs=CRS
    )
    # find
    field = gc.patch.field_for(catalog.query(site), site)
    # compute
    patcher = gp.SpatialPatcher(
        geometry=gp.spatial.geometry.Rectangular(size=(128, 128), boundary="pad"),
        sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),
        window=gp.spatial.window.Hann(),
        aggregation=gp.spatial.aggregation.OverlapAdd(),
    )
    patches = GridSampler(patcher=patcher)(field)
    results = ApplyToChips(operator=NDVI)(patches)
    merge = MergePatches(
        aggregation=gp.spatial.aggregation.OverlapAdd(), domain=field.domain
    )
    values = np.asarray(merge(results))
    # keep: commit, then publish
    product_id = f"ndvi_{job_id}"
    storage = icechunk.local_filesystem_storage(str(STATE / "products.icechunk"))
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")
    array = zarr.create_array(
        session.store, name=product_id, shape=values.shape, chunks=(256, 256), dtype="f4"
    )
    array[:] = values
    snapshot_id = session.commit(f"{product_id}: NDVI {request.start} to {request.end}")
    with database() as db:
        db.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (id) DO UPDATE SET snapshot_id = excluded.snapshot_id",
            (product_id, str(request.start), str(request.end), *request.bounds,
             f"products.icechunk#{product_id}", snapshot_id, RECIPE_HASH),
        )
    return product_id


# === app ===
@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # once, at start-up: the tables, the catalog of the archive, one job worker
    STATE.mkdir(exist_ok=True)
    create_tables()
    files = sorted(glob.glob(str(ARCHIVE / "*.tif")))
    app.state.catalog = build_raster_catalog(
        files, filename_regex=r"_(?P<date>\d{8})", date_format="%Y%m%d"
    )
    app.state.workers = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    yield
    app.state.workers.shutdown()


app = fastapi.FastAPI(title="GeoStack demo backend", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# === search ===
@app.get("/products")
def search_products(
    bbox: str | None = None, start: datetime.date | None = None,
    end: datetime.date | None = None,
) -> list[Product]:
    """Our products that touch a box and a period: a query on our own table."""
    clauses, values = ["1 = 1"], []
    if bbox is not None:
        x0, y0, x1, y1 = (float(part) for part in bbox.split(","))
        clauses.append("x0 <= ? AND x1 >= ? AND y0 <= ? AND y1 >= ?")
        values += [x1, x0, y1, y0]
    if start is not None:
        clauses.append("end_time >= ?")
        values.append(str(start))
    if end is not None:
        clauses.append("start_time <= ?")
        values.append(str(end))
    query = f"SELECT * FROM products WHERE {' AND '.join(clauses)} ORDER BY id"
    with database() as db:
        return [product_from_row(row) for row in db.execute(query, values)]


@app.get("/products/{product_id}")
def get_product(product_id: str) -> Product:
    with database() as db:
        row = db.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="no such product")
    return product_from_row(row)


# === jobs ===
@app.post("/jobs", status_code=202)
def start_job(
    request: RunRequest, http: fastapi.Request, response: fastapi.Response
) -> Job:
    """Accept a run and answer at once; the same request always gives the same job."""
    # job id = SHA-256(request ‖ recipe hash): a retry cannot start a second run
    payload = request.model_dump_json() + RECIPE_HASH
    job_id = hashlib.sha256(payload.encode()).hexdigest()[:12]
    with database() as db:
        # one statement decides who starts the job: a new id is inserted, a failed job
        # is queued again, anything else is left alone; concurrent retries cannot both win
        claimed = db.execute(
            "INSERT INTO jobs VALUES (?, 'queued', NULL) ON CONFLICT (id) DO UPDATE"
            " SET status = 'queued' WHERE jobs.status = 'failed'",
            (job_id,),
        )
        started = claimed.rowcount == 1
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not started:
        response.status_code = 200
        return job_from_row(row)
    # only the request that claimed the row queues the work
    http.app.state.workers.submit(run_recipe, job_id, request, http.app.state.catalog)
    return job_from_row(row)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    with database() as db:
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="no such job")
    return job_from_row(row)


# === trap ===
def read_window() -> dict[str, float]:
    """Stand-in for a 200 ms blocking read (GDAL, NumPy, a database driver)."""
    time.sleep(0.2)
    return {"mean": 0.41}


@app.get("/read/blocking")
async def read_blocking() -> dict[str, float]:
    # async def + a blocking call: the event loop stops for every client
    return read_window()


@app.get("/read/plain")
def read_plain() -> dict[str, float]:
    # plain def: FastAPI runs it on a worker thread
    return read_window()


@app.get("/read/offloaded")
async def read_offloaded() -> dict[str, float]:
    # async def that hands the blocking call to a thread
    return await asyncio.to_thread(read_window)


# === end ===
