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
# # Report 13 · The state catalog, run
#
# Regenerates every number in `docs/reports/13_state_catalog.md`: a synthetic
# forecast archive (stages 1 to 5) and a three-model CMIP-style archive
# (stage 6), indexed with today's geocatalog and scored with today's xrtoolz.
# Both archives are written from fixed seeds into a temporary directory. The
# last cell of each part asserts the values the page shows, to the precision it
# shows them.
#
# Lines marked `GLUE` are what the report's workstreams remove. Pinned
# revisions: geocatalog 0.5.1 (geotoolz `088475e`), xrtoolz 0.0.5 (`308c7f2`).
#
# > Jupytext percent-format script; `pixi run notebooks-to-ipynb` makes a
# > notebook of it. It needs geocatalog, xrtoolz, xskillscore and h5netcdf.

# %%
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from geocatalog import GeoSlice
from geocatalog.build import build_xarray_catalog
from geocatalog.load import load_xarray
from geocatalog.storage import from_geoparquet, to_geoparquet
from xrtoolz.interpolate import RegridConservative
from xrtoolz.metrics import CRPS, RMSE, SkillByLeadTime

root = Path(tempfile.mkdtemp(prefix="report13_"))


def values(x: xr.Dataset | xr.DataArray) -> np.ndarray:
    return np.asarray(x["xch4"] if isinstance(x, xr.Dataset) else x)


# %% [markdown]
# ## Part 1 · The regional archive (stages 1 to 5)
#
# Five daily analyses of a column-methane anomaly on a 4 × 4 grid, and three
# two-member forecast runs with leads of 0, 1 and 2 days. The truth is a random
# walk; analysis noise has σ = 0.05 ppb and the forecast error σ_τ = 0.1 + 0.5τ.

# %%
d = root / "states"
d.mkdir()
rng = np.random.default_rng(0)
lat = np.linspace(31.0, 34.0, 4)
lon = np.linspace(-104.0, -101.0, 4)
days = pd.date_range("2026-01-01", periods=5, freq="D")
truth = rng.normal(size=(5, 4, 4)).cumsum(axis=0)  # (valid day, lat, lon)
H = "sha256:3f2a"  # producer hash (Run.pipeline_hash in practice)


def write(path: Path, data: np.ndarray, times, **attrs) -> str:
    ds = xr.Dataset(
        {"xch4": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={k: str(v) for k, v in attrs.items()},
    )
    ds.to_netcdf(path, engine="h5netcdf")
    return str(path)


# analyses: one per day; parent = the previous run's member-0 forecast
for i, t in enumerate(days):
    parent = "" if i == 0 else str(d / f"fc_{days[i - 1]:%Y%m%d}_m0.nc")
    a = truth[i] + 0.05 * rng.normal(size=(4, 4))
    write(d / f"ana_{t:%Y%m%d}.nc", a[None], [t], kind="analysis",
          run_time=t, member=-1, producer=H, parent_uri=parent)
# forecasts: runs on days 0..2, 2 members, leads 0..2 days
for i, t in enumerate(days[:3]):
    for m in range(2):
        sigma = (0.1 + 0.5 * np.arange(3))[:, None, None]
        err = rng.normal(size=(3, 4, 4)) * sigma
        write(d / f"fc_{t:%Y%m%d}_m{m}.nc", truth[i : i + 3] + err,
              days[i : i + 3], kind="forecast", run_time=t, member=m,
              producer=H, parent_uri=d / f"ana_{t:%Y%m%d}.nc")

# %% [markdown]
# ### Stage 1 · Index

# %%
cat = build_xarray_catalog(sorted(d.glob("*.nc")), crs="EPSG:4326")
n_built = len(cat.gdf)
# GLUE: copy the state attributes into extra columns
attrs = [xr.open_dataset(p, engine="h5netcdf").attrs for p in cat.gdf.filepath]
cat.gdf["kind"] = [a["kind"] for a in attrs]
cat.gdf["run_time"] = pd.to_datetime([a["run_time"] for a in attrs])
cat.gdf["member"] = [int(a["member"]) for a in attrs]
cat.gdf["producer"] = [a["producer"] for a in attrs]
cat.gdf["parent_uri"] = [a["parent_uri"] for a in attrs]
to_geoparquet(cat, d / "states.parquet")
cat = from_geoparquet(d / "states.parquet")
dtypes = cat.gdf.dtypes
n_built, str(dtypes["run_time"]), str(dtypes["member"])

# %% [markdown]
# ### Stage 2 · Queries as `where` strings

# %%
eq = cat.where("kind == 'forecast' and run_time == '2026-01-02' and member == 0")
chain = cat.where("kind == 'forecast' and member == 0"
                  " and run_time >= '2026-01-02' and run_time < '2026-01-03'")
ana = cat.where("kind == 'analysis'").gdf
ana = ana[ana.index.right <= pd.Timestamp("2026-01-03 06:00")]
latest = Path(ana.iloc[ana.index.right.argmax()].filepath).name
ens = cat.where("kind == 'forecast' and run_time < '2026-01-02'")
ens_members = sorted(ens.gdf.member.tolist())
len(eq.gdf), [Path(p).name for p in chain.gdf.filepath], latest, ens_members

# %% [markdown]
# ### Stage 3 · Overlapping runs: `load_xarray` keeps one

# %%
jan3 = GeoSlice(bounds=(-104.5, 30.5, -100.5, 34.5),
                interval=pd.Interval(days[2], days[2], closed="both"),
                resolution=(1.0, 1.0), crs="EPSG:4326")
m0 = cat.where("kind == 'forecast' and member == 0").query(jan3)
n_rows_jan3, n_loaded = len(m0.gdf), load_xarray(m0, jan3).sizes["time"]
n_rows_jan3, n_loaded

# %% [markdown]
# ### Stage 4 · Stack, align and score

# %%
# GLUE: stack (run, member) files into a forecast cube
fc_rows = cat.where("kind == 'forecast'").gdf
pieces = []
indexed = fc_rows.set_index(["run_time", "member"]).sort_index()
for (rt, m), row in indexed.iterrows():
    ds = xr.open_dataset(row.filepath, engine="h5netcdf").load()
    lead = ds.time.values - rt.to_datetime64()
    ds = ds.assign_coords(lead_time=("time", lead))
    ds = ds.swap_dims(time="lead_time").drop_vars("time")
    pieces.append(ds.expand_dims(init_time=[rt], member=[m]))
fc = xr.combine_by_coords(pieces, combine_attrs="drop")
fc = fc.assign_coords(valid_time=fc.init_time + fc.lead_time)

all_days = GeoSlice(bounds=jan3.bounds,
                    interval=pd.Interval(days[0], days[-1], closed="both"),
                    resolution=(1.0, 1.0), crs="EPSG:4326")
an = load_xarray(cat.where("kind == 'analysis'"), all_days)
ref = an.sel(time=fc.valid_time).drop_vars("time")  # GLUE: init + lead → valid

dims = ("init_time", "lat", "lon")
rmse = values(SkillByLeadTime(RMSE("xch4", dims=dims))(fc.mean("member"), ref))
crps = values(SkillByLeadTime(CRPS("xch4", ensemble_dim="member", dims=dims))(fc, ref))
try:
    SkillByLeadTime(RMSE("xch4", dims=("member", *dims)))(fc, ref)
    member_less = "ok"
except ValueError as e:
    member_less = f"ValueError: {e}"
bcast = values(SkillByLeadTime(RMSE("xch4", dims=("member", *dims)))(
    *xr.broadcast(fc, ref)))
sig = 0.1 + 0.5 * np.arange(3)
expected_mean = np.sqrt(sig**2 / 2 + 0.05**2)  # 2-member mean error
expected_member = np.sqrt(sig**2 + 0.05**2)
dict(fc.sizes), rmse.round(3), crps.round(3), bcast.round(3), member_less

# %% [markdown]
# ### Stage 5 · Lineage

# %%
# GLUE: lineage is a walk over parent_uri (one parent per state here)
by_uri = cat.gdf.set_index("filepath")
uri, lineage = str(d / "ana_20260104.nc"), []
while uri:
    lineage.append(Path(uri).stem)
    uri = by_uri.loc[uri, "parent_uri"]
" ← ".join(lineage)

# %%
assert n_built == 11
assert str(dtypes["run_time"]).startswith("datetime64")
assert str(dtypes["member"]) == "int64"
assert len(eq.gdf) == 0  # the string == on a datetime column misses
assert [Path(p).name for p in chain.gdf.filepath] == ["fc_20260102_m0.nc"]
assert latest == "ana_20260103.nc"
assert ens_members == [0, 1]
assert (n_rows_jan3, n_loaded) == (3, 1)
assert dict(fc.sizes) == {"init_time": 3, "member": 2, "lead_time": 3,
                          "lat": 4, "lon": 4}
np.testing.assert_allclose(rmse.round(3), [0.076, 0.467, 0.746])
np.testing.assert_allclose(crps.round(3), [0.048, 0.356, 0.605])
np.testing.assert_allclose(bcast.round(3), [0.109, 0.617, 1.132])
np.testing.assert_allclose(expected_mean.round(3), [0.087, 0.427, 0.779])
np.testing.assert_allclose(expected_member.round(3), [0.112, 0.602, 1.101])
assert member_less.startswith("ValueError")
assert lineage == ["ana_20260104", "fc_20260103_m0", "ana_20260103",
                   "fc_20260102_m0", "ana_20260102", "fc_20260101_m0",
                   "ana_20260101"]

# %% [markdown]
# ## Part 2 · Three models on three grids (stage 6)
#
# Models A (1°, 3 members), B (2.5°, 1) and C (2°, 2), 90 days of daily
# near-surface temperature each, with a model bias and cell-scale weather.

# %%
d6 = root / "cmip"
d6.mkdir()
T = 90
days6 = pd.date_range("2026-06-01", periods=T, freq="D")
MODELS = {"A": (1.0, 0.0, 3), "B": (2.5, 1.5, 1), "C": (2.0, -0.7, 2)}  # Δ, bias, N


def grid(res: float) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(-90 + res / 2, 90, res), np.arange(res / 2, 360, res)


def run(res: float, bias: float, seed: int) -> xr.Dataset:
    la, lo = grid(res)
    r = np.random.default_rng(seed)
    clim = 288 - 30 * np.sin(np.deg2rad(la))[:, None] ** 2 + bias + 0 * lo
    tas = clim + r.normal(0, 2.0, (T, la.size, lo.size))
    return xr.Dataset({"tas": (("time", "lat", "lon"), tas)},
                      coords={"time": days6, "lat": la, "lon": lo})


for s, (res, bias, n) in MODELS.items():
    for m in range(n):
        ds = run(res, bias, seed=10 * ord(s) + m)
        ds.attrs = {"source_id": s, "member_id": f"r{m + 1}i1p1f1",
                    "grid_res": res}
        ds.to_netcdf(d6 / f"tas_{s}_r{m + 1}.nc", engine="h5netcdf")

cat6 = build_xarray_catalog(sorted(d6.glob("tas_*.nc")), crs="EPSG:4326")
attrs6 = [xr.open_dataset(p, engine="h5netcdf").attrs for p in cat6.gdf.filepath]
cat6.gdf["producer"] = [a["source_id"] for a in attrs6]  # GLUE
cat6.gdf["member"] = [a["member_id"] for a in attrs6]  # GLUE: variant label

# GLUE: one DataTree leaf per producer, members stacked inside it
leaves = {}
for s, rows in cat6.gdf.groupby("producer"):
    runs = [xr.open_dataset(p, engine="h5netcdf").load() for p in rows.filepath]
    leaves[f"/{s}"] = xr.concat(runs, dim="member").assign_coords(
        member=rows.member.values)
tree = xr.DataTree.from_dict(leaves)


def area_mean(ds):  # exact spherical cell areas on a regular grid
    res = float(ds.lat[1] - ds.lat[0])
    w = np.sin(np.deg2rad(ds.lat + res / 2)) - np.sin(np.deg2rad(ds.lat - res / 2))
    return ds.weighted(w).mean(("lat", "lon"))


def leaf(t, s):
    return t[s].to_dataset()


def p99(ds):
    return ds.tas.quantile(0.99, dim="time").mean("member").drop_vars("quantile")


la, lo = grid(2.5)
regrid = RegridConservative(xr.Dataset(coords={"lat": la, "lon": lo}),
                            mode="mean", normalize="destarea")
common = regrid(tree)

# (a) a linear statistic commutes with conservative regridding
lin_native = {s: float(area_mean(leaf(tree, s).tas.mean(("time", "member"))))
              for s in "ABC"}
lin_common = {s: float(area_mean(leaf(common, s).tas.mean(("time", "member"))))
              for s in "ABC"}
# (b) a nonlinear one does not: per-cell p99, then the area mean
p_native = {s: float(area_mean(p99(leaf(tree, s)))) for s in "ABC"}
p_first = {s: float(area_mean(p99(leaf(common, s)))) for s in "ABC"}
p_hybrid = {s: float(area_mean(regrid(p99(leaf(tree, s)).to_dataset(name="tas")).tas))
            for s in "ABC"}
# (c) multi-model mean: one vote per model vs one vote per member
n_mem = {s: MODELS[s][2] for s in "ABC"}
per_model = float(np.mean([lin_native[s] for s in "ABC"]))
per_member = sum(n_mem[s] * lin_native[s] for s in "ABC") / sum(n_mem.values())
lin_native, p_first, per_model, per_member

# %%
assert len(cat6.gdf) == 6
assert sorted(cat6.gdf.member) == sorted(
    ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r1i1p1f1", "r1i1p1f1", "r2i1p1f1"])
assert {s: tree[s].sizes["member"] for s in "ABC"} == {"A": 3, "B": 1, "C": 2}
assert {k: round(v, 4) for k, v in lin_native.items()} == {
    "A": 277.9999, "B": 279.4971, "C": 277.2991}
assert max(abs(lin_native[k] - lin_common[k]) for k in "ABC") < 1e-10
assert {k: round(p_native[k] - lin_native[k], 3) for k in "ABC"} == {
    "A": 4.289, "B": 4.279, "C": 4.288}
assert {k: round(p_first[k] - lin_native[k], 3) for k in "ABC"} == {
    "A": 1.544, "B": 4.279, "C": 2.564}
assert max(abs(p_hybrid[k] - p_native[k]) for k in "ABC") < 1e-10
assert (round(per_model, 4), round(per_member, 4)) == (278.2654, 278.0159)
