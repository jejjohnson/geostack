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
# # Background · the geo pipeline: ten stages on a small archive
#
# Regenerates every measured number on
# `docs/design/pages/geo-pipeline.html` (stub: `docs/design/geo_pipeline.md`).
# Writes a 24-scene synthetic archive, then runs search, catalog, save, stream, split,
# apply, combine and write with geotoolz, and commit and publish with Icechunk and a
# Parquet row.
#
# Needs: geotoolz (geoarray, geocatalog, geocloud, geoops, geopatcher), pipekit-ops, icechunk, zarr, pyarrow, pandas, pyproj.
# It illustrates a docs page and is not a stage of the DVC pipeline; run it by hand:
#
#     python notebooks/background/geo_pipeline.py
#
# It writes `docs/images/background_geo_pipeline/results.json`, the file the page's
# numbers were taken from. Timings depend on the machine and change from run to run; counts, sizes and errors do not. Snapshot ids differ on every run. Under a minute.

# %%
"""The ten-stage pipeline on a small local archive: NDVI over one site.

Stages 1 to 8 use geotoolz as it ships; stages 9 and 10 (commit, publish) use
Icechunk and a GeoParquet table directly, as stand-ins for the proposed APIs.
"""

from __future__ import annotations

import glob
import json
import pathlib
import hashlib
import shutil
import time

import geoarray
import geocatalog as gc
import geocloud.cog
import geoops as gz
import geopatcher as gp
import icechunk
import numpy as np
import pandas as pd
import pipekit_ops
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import zarr
from geocatalog.build import build_raster_catalog
from geoops.patch_ops import ApplyToChips, GridSampler, MergePatches

import tempfile

HERE = pathlib.Path(tempfile.mkdtemp())  # the archive and the work folder are scratch
ARCHIVE = HERE / "archive"
WORK = HERE / "work"
CRS = pyproj.CRS.from_epsg(32630)
PIXEL_M = 10.0


def make_archive() -> None:
    """24 synthetic two-band scenes: 4 tiles × 6 dates, 600 × 600 px at 10 m."""
    shutil.rmtree(ARCHIVE, ignore_errors=True)
    ARCHIVE.mkdir()
    rng = np.random.default_rng(0)
    dates = ["20260605", "20260612", "20260621", "20260702", "20260714", "20260520"]
    origins = {"A": (720000.0, 4380000.0), "B": (726000.0, 4380000.0),
               "C": (720000.0, 4374000.0), "D": (726000.0, 4374000.0)}
    yy, xx = np.mgrid[:600, :600]
    for tile, (x0, y0) in origins.items():
        for d, date in enumerate(dates):
            vegetation = 0.5 + 0.4 * np.sin(xx / 90.0 + d) * np.cos(yy / 70.0)
            red = 1500 - 900 * vegetation + rng.normal(0, 20, (600, 600))
            nir = 1500 + 2200 * vegetation + rng.normal(0, 20, (600, 600))
            data = np.stack([red, nir]).astype(np.uint16)
            transform = (PIXEL_M, 0.0, x0, 0.0, -PIXEL_M, y0)
            scene = geoarray.GeoArray(
                data, np.asarray(transform), CRS, fill_value=0, dims=("band", "y", "x")
            )
            geocloud.cog.write_cog(scene, ARCHIVE / f"S2_{tile}_{date}.tif")


# %%
# Where the recorded output goes. Set BACKGROUND_OUT to write somewhere else.
import os as _os
import pathlib as _pathlib

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(_os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_geo_pipeline"))
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    make_archive()
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir()
    clock: dict[str, float] = {}
    mark = time.perf_counter()

    def lap(stage: str) -> None:
        global mark
        now = time.perf_counter()
        clock[stage] = round((now - mark) * 1e3, 1)
        mark = now

    # Shapes: T = 2 bands; H × W = 400 × 500 px (the site); h × w = 128 × 128 px

    # === search ===
    # the archive: here a folder; in production a STAC, CMR or GEE search
    files = sorted(glob.glob(str(ARCHIVE / "*.tif")))             # → 24 paths
    lap("search")

    # === catalog ===
    # today (geocatalog)
    # index once: footprint, CRS, resolution and time per file; no pixels read
    catalog = build_raster_catalog(                               # 24 paths → 24 rows
        files, filename_regex=r"_(?P<date>\d{8})", date_format="%Y%m%d"
    )
    # today (geocatalog)
    # the request: bounds × time × resolution × CRS
    site = gc.GeoSlice(
        bounds=(723000.0, 4375000.0, 728000.0, 4379000.0),
        interval=pd.Interval(
            pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-30"), closed="both"
        ),
        resolution=(PIXEL_M, PIXEL_M),
        crs=CRS,
    )
    # today (geocatalog)
    hits = catalog.query(site)                                    # 24 rows → 6 rows
    lap("catalog")

    # === save ===
    # today (geocatalog)
    # fetch only the hits next to the compute; local files stay where they are
    staged = gc.staging.stage(hits, dest=str(WORK / "cache"))     # 6 rows → 6 rows
    lap("save")

    # === stream ===
    # today (geocatalog)
    # a Field: the six files as one lazy mosaic on the site's grid
    field = gc.patch.field_for(staged, site)                      # 6 rows → (T, H, W)
    lap("stream")

    # === split ===
    # today (geopatcher)
    patcher = gp.SpatialPatcher(
        geometry=gp.spatial.geometry.Rectangular(size=(128, 128), boundary="pad"),
        sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),
        window=gp.spatial.window.Hann(),
        aggregation=gp.spatial.aggregation.OverlapAdd(),
    )
    # today (geoops)
    patches = GridSampler(patcher=patcher)(field)                 # (T, H, W) → 20 × (T, h, w)
    lap("split")

    # === apply ===
    # today (geoops)
    # NDVI = (ρ_nir − ρ_red) / (ρ_nir + ρ_red)
    ndvi = gz.DNToReflectance(scale=1e-4) | gz.NDVI(red=0, nir=1)
    # today (geoops)
    results = ApplyToChips(operator=ndvi)(patches)                # 20 × (T, h, w) → 20 × (h, w)
    lap("apply")

    # === combine ===
    # today (geoops)
    # window-weighted merge of the overlaps, back on the site's grid
    merge = MergePatches(
        aggregation=gp.spatial.aggregation.OverlapAdd(), domain=field.domain
    )
    product = merge(results)                                      # 20 × (h, w) → (H, W)
    lap("combine")

    # === write ===
    # today (geoops)
    gz.WriteCOG(path=str(WORK / "ndvi_june.tif"))(product)        # (H, W) → 1 file
    lap("write")

    # === commit ===
    # today (icechunk)
    # the same product into a versioned store: visible all at once, or not at all
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(WORK / "products.icechunk"))
    )
    session = repo.writable_session("main")
    values = np.asarray(product)                                  # (H, W)
    array = zarr.create_array(
        session.store, name="ndvi", shape=values.shape, chunks=(256, 256), dtype="f4"
    )
    array[:] = values
    array.attrs["transform"] = [float(v) for v in product.transform]
    array.attrs["crs"] = "EPSG:32630"
    snapshot_id = session.commit("NDVI, site, June 2026")         # → snapshot id
    lap("commit")

    # === publish ===
    # today (pipekit)
    # h = SHA-256(stable_json(state) ‖ 0x00 ‖ weights): the recipe's id
    recipe_hash = pipekit_ops.pipeline_hash(ndvi, b"")
    # one row in the product table: what, where, when, from what, by what
    record = {
        "id": "ndvi_site_2026-06",
        "start_time": "2026-06-01",
        "end_time": "2026-06-30",
        "bbox": list(site.bounds),
        "crs": "EPSG:32630",
        "href": "products.icechunk",
        "snapshot_id": snapshot_id,
        "recipe_hash": recipe_hash,
        "sources": [pathlib.Path(path).name for path in hits.gdf["filepath"]],
    }
    pq.write_table(pa.Table.from_pylist([record]), WORK / "products.parquet")
    lap("publish")

    # === loop ===
    # the next pipeline's search is a query on the product table
    found = pq.read_table(WORK / "products.parquet").to_pylist()[0]
    reopened = repo.readonly_session(snapshot_id=found["snapshot_id"])
    again = zarr.open_array(reopened.store, path="ndvi")[:]       # → (H, W)
    # === end ===

    whole = ndvi(field.reader.load())
    R = dict(
        clock=clock,
        archive_files=len(files),
        archive_mb=round(sum(pathlib.Path(f).stat().st_size for f in files) / 1e6, 1),
        catalog_rows=len(catalog), hits=len(hits), staged=len(staged),
        hits_mb=round(sum(pathlib.Path(f).stat().st_size for f in hits.gdf["filepath"]) / 1e6, 1),
        field_shape=list(field.reader.shape), patches=len(patches),
        out_shape=list(values.shape),
        max_abs_diff=float(np.nanmax(np.abs(np.asarray(whole) - values))),
        ndvi_mean=float(np.nanmean(values)),
        cog_kb=round((WORK / "ndvi_june.tif").stat().st_size / 1e3, 1),
        snapshot_id=snapshot_id, recipe_hash=recipe_hash, record=record,
        roundtrip=bool(np.allclose(again, values.astype("f4"), equal_nan=True)),
        recipe_json=json.loads(pipekit_ops.dumps(ndvi)),
        lines=sum(1 for _ in open(__file__)),
    )
    print(json.dumps({k: v for k, v in R.items() if k != "recipe_json"}, indent=1, default=str))
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=str) + "\n")
