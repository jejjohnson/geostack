---
title: "GeoStack vision: case studies"
subtitle: "Design · thirteen real workflows run through the levels and run modes"
short_title: Case studies
description: "Thirteen workflows from the stack's real use, from a methane plume detector to a labelling app, each run through the vision's levels and run modes, with the gaps found and the findings CS-F1 to CS-F17 that change the vision."
date: 2026-10-23
keywords:
  - case studies
  - vision
  - levels
  - run modes
  - methane
  - 4D-Var
  - GeoBench
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

Thirteen workflows from the stack's real use, spread across the whole quadrant, each run through the vision's levels and run modes to find where it holds and where it needs changing.
Companion to [the vision](./vision.md).

:::{note} Names
These cases were written before the [esmax](./esmax.md) plan; they use today's package name, somax, for the ocean model.
Under the plan, `somax.geo` becomes `esmax_core.geo` and `somax.ShallowWater` becomes `esmax_ocean.ShallowWater`.
Findings are numbered CS-F1 to CS-F17 and are listed with every other decision on [Decisions](./decisions.md).
:::

## How to read the code

The code shows the **target state**: the vision with every finding applied.
Each line that depends on a finding is tagged `# ← CS-F3`, so the gap under each case says what is missing today and the code shows what it looks like once fixed.

| Annotation | Means | Example |
| --- | --- | --- |
| `Float[Array, "P C H W"]` | a raw array (jaxtyping); letters are sizes | `logits: Float[Array, "P H W"]` |
| `GeoArray["patch band y x"]` | a geoarray carrier; the string names its dims, jaxtyping-style | `alpha: GeoArray["patch y x"]` |
| `GeoSwath["band row col"]`, `GeoPoints["time variable obs"]` | the swath and point carriers, same style | `stations: GeoPoints["time variable obs"]` |
| `DataArray["time depth lat lon"]`, `Dataset["time lat lon"]` | xarray carriers, same style; checkable once CS-F1 lands | `mld: DataArray["time lat lon"]` |
| `GeoImage`, `GeoTracks`, `GeoImageSeries`, `Detections` | readable aliases, used in L3 signatures | `GeoImageSeries` ≡ `GeoArray["time y x"]` on any carrier (CS-F1) |

Math comments use ∑, ∇, δ and friends; every symbol is defined where it first appears.
Reductions and contractions on arrays are written with einx (`einx.dot`, `einx.mean`, `einx.rearrange`); linear solves use lineax.
One step per line: each intermediate is named, typed and commented, and no call is nested inside another call's arguments.

## At a glance

The thirteen cases cover all four quadrants; seven still hand-write how their steps run, so the computation side of the vision is ahead of the orchestration side.

```text
top right: built-in computation, packaged orchestration
│   1 plume detector · 7 daily alerts (far right) · 13 labelling app
│
top left: your own computation, packaged orchestration
│   5 new retrieval method · 9 GeoBench submission
│
top centre
│   12 ensemble forecast
│
centre
│   10 flaring graph
│
bottom right: built-in computation, your own orchestration
│   2 emission rate · 3 SLA 4D-Var · 4 MLD climatology · 6 AEMET validation
│
bottom left: your own computation and orchestration
    8 WENO5 scheme (far left) · 11 GP interpolation of SST
```

Positions are qualitative: right means the user calls built-in operators or algorithms, up means a packaged pipeline or runner orchestrates the steps.

| Case | Quadrant | Levels used | Run modes | Readable types | Main finding |
| --- | --- | --- | --- | --- | --- |
| 1 Plume detector training | top right | L3 ETL, L2 features, L1 in the loss | eager, compiled | `GeoImage`, `GeoPatches` | concurrent I/O has no owner (CS-F4) |
| 2 Emission rate | bottom right | L3 algorithm over L1 | eager, lazy | `GeoSwathImage`, `GeoPoints` | two carrier families in one call (CS-F2); records (CS-F6) |
| 3 SLA 4D-Var | bottom right | L3 algorithm, hand-written cycle | lazy, eager, compiled | `GeoMap`, `GeoTracks`, `GeoImageSeries` | padding is a crossing (CS-F3); cycling (CS-F4) |
| 4 MLD climatology | bottom right | L2 kernels, native xarray | lazy | none available | readable types stop at `GeoArray` (CS-F1) |
| 5 New retrieval method | top left | L3 → L1 → L3 | eager | `GeoSwathImage` | angles are derived geometry (CS-F5); fitted slots (CS-F9) |
| 6 AEMET validation | bottom right | L2, short custom graph | lazy, eager | `GeoTimeSeries` | samplers agree; tables (CS-F10) |
| 7 Daily alert pipeline | top right, far | L3 only | eager, concurrent | `Detections` | versions, provenance, idempotent runs (CS-F11) |
| 8 WENO5 scheme | bottom left, far | L1 only | compiled | none, by choice | carrier-free geometry helpers; the way up (CS-F12) |
| 9 GeoBench submission | top left | packaged runner, user model at any level | compiled | `GeoTracks`, `GeoImageSeries` as the contract | types must be checkable (CS-F1); runners own splits (CS-F13) |
| 10 Flaring graph | centre | L1 kernel, `KernelOp`, custom graph | eager | `GeoImage`, `Detections` | promotion from graph to pipeline (CS-F14) |
| 11 GP interpolation of SST | bottom left | own model at L1, own loop | compiled | `GeoPoints`, `GeoArray["moment y x"]` | a convention for uncertainty (CS-F15) |
| 12 Ensemble forecast | top centre | packaged runner over a model | compiled, lazy | `GeoMap`, member cube | algorithms expose their L1 core; runners own seeds (CS-F16) |
| 13 Labelling app | right, upper | L2 operators in a web handler | eager, compiled | `GeoImage`, `GeoMap` | latency: imports and compile ahead of time (CS-F17) |

## Case 1: methane plume detector, ETL to training

**Who and goal.**
An ML engineer trains a plume detector on Sentinel-2 over oil and gas basins: labelled scenes in, a trained model and validation metrics out.

**Where it sits.**
L3 for the ETL (packaged), L2 for the model's features, compiled training.
Mostly built-in runners and built-in operators: top right of the quadrant.

| Stage | Level | Run mode | Carrier in → out |
| --- | --- | --- | --- |
| query catalog, read scenes | L3 pipeline | eager, concurrent I/O | catalog rows → `GeoImage` per scene |
| cloud mask, rasterize labelled plume polygons | L2 operators | eager | `GeoImage` + polygons → `GeoImage` + `GeoMap` label |
| cut 256 × 256 patches | L2 (geopatcher) | eager | → `GeoPatches` (patch, band, y, x) |
| matched-filter feature + train | L1 kernel inside the loss | compiled | raw arrays (64, C, 256, 256) |
| validate, write metrics | L2 + pipekit | eager | → metrics JSON |

What `plume_training_set` does inside:

```python
def plume_training_set(
    catalog: Catalog,
    basins: list[str],
    labels: str,
    patch: int,
    max_cloud: float,
    runner: Runner,  # ← CS-F4: the runner owns concurrent I/O
) -> tuple[GeoArray["patch band y x"], GeoArray["patch y x"]]:
    # the basins' outlines, as one AOI
    aoi: AOI = geocatalog.aois(basins)
    # catalog rows for every Sentinel-2 L1C scene over them
    rows: CatalogRows = catalog.query("sentinel-2-l1c", aoi=aoi, max_cloud=max_cloud)
    # one read task per scene, joined before patching; 13 bands
    read: Operator = geoproducts.sentinel2.Read()
    scenes: list[GeoArray["band y x"]] = runner.map(read, rows)
    # plume outlines
    polygons: gpd.GeoDataFrame = geopandas.read_parquet(labels)

    cloud_mask: Operator = geoops.CloudMask()
    xs: list[GeoArray["patch band y x"]] = []
    ys: list[GeoArray["patch y x"]] = []
    for scene in scenes:
        # 𝟙[clear](y, x)
        clear: GeoArray["y x"] = ~cloud_mask(scene)
        # the outlines that touch this scene
        local: gpd.GeoDataFrame = polygons.clip(scene.footprint)
        # outlines → labels on the scene's grid; eager-only geometry operator ← CS-F8
        rasterize: Operator = geoops.geom.RasterizeLike(scene)
        label: GeoArray["y x"] = rasterize(local)
        # 256 × 256 windows with ≥ 80 % clear pixels: (Pᵢ, 13, 256, 256), transforms (Pᵢ, 6)
        x: GeoArray["patch band y x"] = geopatcher.split(
            scene, size=patch, where=clear, min_valid=0.8
        )
        # the same windows on the labels: (Pᵢ, 256, 256)
        y: GeoArray["patch y x"] = geopatcher.split(label, like=x)
        xs.append(x)
        ys.append(y)

    # (P, 13, 256, 256) and (P, 256, 256)
    x_all: GeoArray["patch band y x"] = geopatcher.concat(xs)
    y_all: GeoArray["patch y x"] = geopatcher.concat(ys)
    return x_all, y_all
```

What the user writes:

```python
# the ETL: a packaged pipeline with a concurrent runner
runner: Runner = pipekit.runners.Concurrent()
patches: GeoArray["patch band y x"]
labels: GeoArray["patch y x"]
patches, labels = pipelines.plume_training_set(
    catalog,
    basins=["permian", "turkmenistan"],
    labels="plumes_v3.parquet",
    patch=256,
    max_cloud=0.2,
    runner=runner,
)

# crossing, once: NumPy → JAX
jp: GeoArray["patch band y x"] = patches.to_jax()
jy: GeoArray["patch y x"] = labels.to_jax()

# the matched filter: fitted on the training patches (μ, Σ⁻¹), then a JaxOperator whose arrays are leaves
mf_spec: geoops.MatchedFilter = geoops.MatchedFilter(target=ch4_spectrum, dim="band")
mf_fitted: geoops.MatchedFilter = mf_spec.fit(patches)
mf: JaxOperator = pipekit_jax.to_jax(mf_fitted)


@eqx.filter_jit
def loss(
    model: eqx.Module, batch: GeoArray["patch band y x"], y: Float[Array, "P H W"]
) -> Float[Array, ""]:
    # α = tᵀΣ⁻¹(x − μ) / tᵀΣ⁻¹t per pixel; t the CH₄ target, μ and Σ the background
    alpha: GeoArray["patch y x"] = mf(batch)
    # [x, α] on the channel axis: C + 1 = 14 channels
    feats: Float[Array, "P C1 H W"] = einx.rearrange(
        "p c h w, p h w -> p (c + 1) h w", batch.data, alpha.data
    )
    # logits = f_θ(feats)
    logits: Float[Array, "P H W"] = model(feats)
    # BCE(σ(logits), y) per pixel
    bce: Float[Array, "P H W"] = optax.sigmoid_binary_cross_entropy(logits, y)
    # ℒ = (1 / P·H·W) ∑ BCE
    return einx.mean("[p h w]", bce)
```

**What it stresses.**

- **Holds:** one crossing (`to_jax()`); I/O never under JAX; the fitted matched filter becomes a `JaxOperator` with μ and Σ⁻¹ as leaves.
- **Gap:** "read concurrently" has no owner yet.
  The pipeline needs a runner that fans out I/O stages and joins before patching: the missing pipelines page (vision V4).
- **Gap:** labels arrive as polygons (GeoParquet); rasterizing them is a geometry operator with no primitive (GDAL), so this stage can never run under JAX.
  That is fine, but the vision should say which operators are eager-only.

## Case 2: emission rate from a detected plume

**Who and goal.**
An analyst quantifies a plume EMIT saw over a compressor station: an enhancement map and ERA5 winds in, an emission rate with an uncertainty out.

**Where it sits.**
An L3 algorithm in plumax, run eagerly, calling an L1 primitive.
It mixes two carrier families in one call: the plume is a `GeoSwathImage`, the winds an xarray `Dataset`.

```python
# L1, plumax/_src/array.py: integrated mass enhancement, arrays only
def integrated_mass_enhancement(
    dx_ch4: Float[Array, "N"],  # ΔXCH₄ in the N plume pixels, ppb
    area: Float[Array, "N"],  # pixel areas, m² (derived geometry)
    column_air: float,  # N_air, dry-air column, mol m⁻²
) -> Float[Array, ""]:
    # ∑ᵢ ΔXᵢ Aᵢ, ppb m²
    weighted: Float[Array, ""] = einx.dot("n, n ->", dx_ch4, area)
    # IME = M_CH₄ · N_air · ∑ᵢ ΔXᵢ Aᵢ · 10⁻⁹, kg; M_CH₄ = 0.016 kg mol⁻¹
    return M_CH4 * column_air * weighted * 1e-9


# L3, plumax/geo/estimate.py: two carrier families in, a record out
def estimate_emission(
    plume: GeoSwath["row col"],  # ΔXCH₄ on the EMIT swath, ppb
    mask: GeoSwath["row col"],  # plume mask, same swath
    winds: Dataset["time lat lon"],  # ERA5 u10, v10, lazy ← CS-F2: a second family
) -> Detections:  # GeoPoints["variable obs"] ← CS-F6
    # derived geometry from the swath's lat/lon leaves, m²
    area: GeoSwath["row col"] = geoarray.pixel_area(plume)
    # the mask as weights m ∈ {0, 1}
    m: Float[Array, "R C"] = mask.data.astype(area.data.dtype)
    # plume area ∑_{r,c} m A, m²
    plume_area: Float[Array, ""] = einx.dot("r c, r c ->", area.data, m)
    # L = √(plume area), m
    length: Float[Array, ""] = jnp.sqrt(plume_area)
    # where to sample the wind
    lon0, lat0 = geoarray.centroid(plume, where=mask)
    # crossing inside the algorithm, by value in xarray first ← CS-F2
    w_lazy: Dataset[""] = winds.interp(time=plume.time, lat=lat0, lon=lon0)
    w: Dataset[""] = w_lazy.compute()
    # U₁₀ = √(u² + v²), m s⁻¹
    u10: float = float(np.hypot(w.u10, w.v10))
    # U_eff = a · U₁₀ + b; a, b empirical
    u_eff: float = A_EFF * u10 + B_EFF
    # the plume pixels only, (N,)
    dx_in: Float[Array, "N"] = plume.data[mask.data]
    area_in: Float[Array, "N"] = area.data[mask.data]
    ime: Float[Array, ""] = integrated_mass_enhancement(dx_in, area_in, COLUMN_AIR)
    # Q = U_eff · IME / L, kg s⁻¹
    q: Float[Array, ""] = u_eff * ime / length
    return Detections.from_record(lon0, lat0, time=plume.time, Q=q, sigma_Q=q * REL_ERR)
```

**What it stresses.**

- **Gap: an L3 algorithm takes carriers from two families.**
  The vision says the carrier chooses the run mode; here two carriers choose two modes in one call.
  The rule needs a line: an L3 algorithm may take several families, and does its own crossings inside, by value in xarray first.
- **Gap: the result is a record, not a raster.**
  An emission estimate is a point with variables (Q, σ_Q, time).
  `GeoPoints["variable obs"]` carries it, but the vision's readable types only name rasters and series; a `GeoRecords` or `Detections` type may be needed.
- **Holds: derived geometry from a swath.**
  `pixel_area` reads the lat/lon leaves and returns arrays; the primitive never sees the swath.

## Case 3: SLA mapping by 4D-Var from along-track altimetry

**Who and goal.**
An ocean researcher maps daily sea-level anomaly over the Gulf Stream for 2025: CMEMS along-track SLA and a DUACS background in, an analysis cube in Zarr out, scored against one held-out satellite.

**Where it sits.**
A cycled assimilation: an eager outer loop over 7-day windows, a compiled inner minimisation, lazy reading and writing at both ends.
Built-in operators, custom orchestration: the left half of the quadrant's top.

```python
# lazy, by value: the DUACS background and the along-track files for 2025
bg_cube: DataArray["time lat lon"] = xrproducts.cmems.open(
    "duacs_l4_sla", region=gulf_stream, year=2025
)
tracks: Dataset["obs"] = xrproducts.cmems.open(
    "l3_alongtrack_sla", region=gulf_stream, year=2025
)


def cycle(
    state: GeoArray["y x"] | None, window: Window
) -> tuple[GeoArray["y x"], GeoArray["time y x"]]:
    # x_b: the previous analysis, or DUACS on the first window ← CS-F4: warm start
    if state is None:
        bg_day: DataArray["lat lon"] = bg_cube.sel(time=window.start)
        x_b: GeoArray["y x"] = GeoArray.from_xarray(bg_day)
    else:
        x_b: GeoArray["y x"] = state
    # this window's tracks, Jason-3 held out
    tracks_w: Dataset["obs"] = tracks.sel(time=window.slice)
    obs: GeoPoints["obs"] = GeoPoints.from_dsg(tracks_w, exclude="j3")
    # crossings: fixed N = 20 000 with valid = False on the pad ← CS-F3
    jobs: GeoPoints["obs"] = obs.to_jax(pad_to=20_000)
    jx_b: GeoArray["y x"] = x_b.to_jax()
    # xₐ = argmin J(x₀), J = ½‖x₀ − x_b‖²/σ_b² + ½‖m ⊙ (y − H(M(x₀)))‖²/σ_o²
    series: GeoArray["time y x"] = vardax.geo.assimilate(
        jx_b, jobs, window.times, sigma_b=0.05, sigma_o=0.03
    )
    # back to NumPy; the next state is the analysis at the window's last time
    series_np: GeoArray["time y x"] = series.to_numpy()
    next_state: GeoArray["y x"] = series_np.isel(time=-1)
    return next_state, series_np


# 52 seven-day windows over 2025
windows: Windows = pipekit.windows("2025-01-01", "2025-12-31", length="7D")
# where each window's series is appended
out: Zarr = Zarr("gulfstream_sla_2025.zarr")
# the runner owns the loop, the state and the append ← CS-F4
runner: Runner = pipekit.runners.Cycling(windows=windows)
run: CyclingRun = runner.run(cycle, init=None, out=out, append_dim="time")

# the analysis cube, lazy
analysis: DataArray["time lat lon"] = xr.open_zarr(run.out).sla
# RMSE = √(𝔼[(ηₐ − y_j3)²]) against the held-out Jason-3 track
```

The 0.05 m and 0.03 m are illustrative σ_b and σ_o; `exclude="j3"` keeps the held-out satellite out of every window.

**What it stresses.**

- **Gap: padding is a crossing.**
  `to_jax()` is not enough for observations; they need a fixed N with a validity mask.
  Make it part of the named crossing: `obs.to_jax(pad_to=N)`.
- **Gap: cycling needs a stateful runner.**
  Operational cycling warm-starts each window from the last analysis, which this loop does not do yet; the hand-written `for` loop here is exactly what a packaged runner should own (vision V4), with the state carried between windows and each window's output appended to Zarr.
- **Holds:** the L3 assimilator never sees xarray; the cube is only touched by value, before and after.

## Case 4: MLD climatology and anomalies from GLORYS

**Who and goal.**
A physical oceanographer computes the mixed-layer depth from 30 years of daily GLORYS12 temperature and salinity, then a monthly climatology and anomalies, for the North Atlantic.

**Where it sits.**
Lazy throughout, larger than memory.
L2 kernel operators (state physics, so xrops per X2) followed by native xarray operations.
Nothing compiles and nothing is a `GeoArray`.

```python
# lazy, daily 1993–2022, dask chunks of 30 days
ts: Dataset["time depth lat lon"] = xrproducts.cmems.open(
    "glorys12_daily", vars=["thetao", "so"], region=north_atlantic
)

# σ₀(T, S) by TEOS-10, elementwise
to_sigma0: Operator = xrops.ocn.Sigma0()
sigma0: DataArray["time depth lat lon"] = to_sigma0(ts)

# consumes depth; z read from the depth coordinate and passed to the kernel as an array
to_mld: Operator = xrops.ocn.MixedLayerDepth(threshold=0.03, dim="depth")
mld: DataArray["time lat lon"] = to_mld(sigma0)

# the readable type matches a lazy carrier too ← CS-F1
assert geoarray.types.matches(mld, GeoImageSeries)

# clim(m, lat, lon) = (1 / |Tₘ|) ∑_{t ∈ Tₘ} MLD(t, lat, lon), Tₘ the days in month m
# native groupby underneath, a named operator on top ← CS-F7
to_clim: Operator = xrops.Climatology(freq="month")
clim: DataArray["month lat lon"] = to_clim(mld)

# anom(t) = MLD(t) − clim(month(t))
to_anom: Operator = xrops.Anomaly(clim)
anom: DataArray["time lat lon"] = to_anom(mld)
anom.to_zarr("na_mld_anomaly.zarr")

# the same as a graph of the named operators, so it can be versioned ← CS-F7, CS-F14
graph: pipekit.Graph = pipekit.Graph(
    sigma0=to_sigma0, mld=to_mld, clim=to_clim, anom=xrops.Anomaly.from_stage("clim")
)
```

**What it stresses.**

- **Gap: readable types stop at the carrier boundary.**
  `mld` is a series of maps, but because it is a `DataArray` the vision gives it no readable type: `GeoImageSeries` is an alias of `GeoArray`.
  A pipeline that takes "a series of maps" cannot say so for its lazy input.
- **Gap: native operations are not pipeline stages.**
  `groupby(...).mean()` is the right call, but the vision says every stage is a named operator.
  Either xrops wraps the handful of native operations pipelines need (`Climatology`, `Anomaly`) or a stage may be a named native call.
- **Holds:** the MLD kernel is the same L1 function that runs on a `GeoVolume`; only the wrapper differs.

## Case 5: a researcher swaps in a new retrieval method

**Who and goal.**
A retrieval scientist tests a column-wise matched filter with an air-mass correction against the packaged EMIT plume pipeline, keeping every other stage identical.

**Where it sits.**
Starts at L3, steps down to L1 for one stage, and comes back up: the vision's "step down in one line" under real use.
EMIT's matched filter is fitted per detector column, because each cross-track column has its own noise.

What `emit_plumes` builds inside:

```python
def emit_plumes(scene_id: str) -> pipekit.Graph:
    # → GeoSwath["band row col"], 285 bands; lat/lon and sun/view angles as leaves
    read: Operator = geoproducts.emit.ReadL1B(scene_id)
    # GeoSwath["band row col"] → GeoSwath["band row col"], 2100–2450 nm only
    window: Operator = geoops.SpectralWindow(2100, 2450, dim="band")
    # t: the unit CH₄ absorption spectrum, (C,)
    target: Float[Array, "C"] = plumes.ch4_unit_spectrum()
    # α = tᵀΣ⁻¹(x − μ) / tᵀΣ⁻¹t, μ and Σ fitted per column → GeoSwath["row col"]
    mf: Operator = geoops.MatchedFilter(target=target, dim="band", fit_per="col")
    # the slot declares that it holds a fitted operator ← CS-F9
    mf_slot: pipekit.Slot = pipekit.Slot(mf, fitted=True)
    # ΔXCH₄ = α · scale, ppb → GeoSwath["row col"]
    to_ppb: Operator = plumes.ScaleToPPB()
    return pipekit.Graph(
        read=read, window=window, matched_filter=mf_slot, to_ppb=to_ppb
    )
```

What the user writes:

```python
# the packaged pipeline as a graph, and its baseline output
pipe: pipekit.Graph = pipelines.emit_plumes.build(
    scene_id="EMIT_L1B_RAD_001_20260714T081512"
)
baseline: GeoSwath["row col"] = pipe.run()  # ΔXCH₄, ppb


# L1: statistics over the R rows of one column, C bands
def mf_amf(
    x: Float[Array, "R C"], t: Float[Array, "C"], amf: Float[Array, "R"]
) -> Float[Array, "R"]:
    # μ = (1/R) ∑ᵣ xᵣ
    mu: Float[Array, "C"] = einx.mean("[r] c", x)
    # x̃ᵣ = xᵣ − μ
    xc: Float[Array, "R C"] = einx.subtract("r c, c", x, mu)
    # Σ = (1/R) ∑ᵣ x̃ᵣ x̃ᵣᵀ
    sigma: Float[Array, "C C"] = einx.dot("r c1, r c2 -> c1 c2", xc, xc) / x.shape[0]
    # w = Σ⁻¹t, solved, never inverted
    op: lx.MatrixLinearOperator = lx.MatrixLinearOperator(
        sigma, lx.positive_semidefinite_tag
    )
    w: Float[Array, "C"] = lx.linear_solve(op, t).value
    # numerator x̃ᵣᵀw and normaliser tᵀw
    num: Float[Array, "R"] = einx.dot("r c, c -> r", xc, w)
    norm: Float[Array, ""] = einx.dot("c, c ->", t, w)
    # αᵣ = x̃ᵣᵀw / (amfᵣ · tᵀw)
    return num / (amf * norm)


# L2: consumes rows and bands, maps over columns; GeoSwath["band row col"] → GeoSwath["row col"]
kernel_op: KernelOp = KernelOp(
    mf_amf,
    "(r,c),(c),(r)->(r)",
    core={"r": "row", "c": "band"},
    geometry={
        "amf": geoarray.air_mass_factor
    },  # AMF = 1/cos θ_sun + 1/cos θ_view, from the carrier ← CS-F5
    fitted=False,  # fits μ, Σ inside; replace() checks the slot ← CS-F9
)
target: Float[Array, "C"] = plumes.ch4_unit_spectrum()
new_stage: KernelOp = kernel_op.bind(t=target)

# the same pipeline with one stage replaced
candidate_pipe: pipekit.Graph = pipe.replace("matched_filter", new_stage)
candidate: GeoSwath["row col"] = candidate_pipe.run()

# score both against controlled releases
report: Comparison = pipekit.evaluate.compare(
    baseline, candidate, truth="controlled_releases_2026.parquet"
)
```

**What it stresses.**

- **Holds:** one stage swapped, the rest identical; the generalized-ufunc spec expresses "statistics over rows, per column" without a loop.
- **Gap: derived geometry is more than spacing.**
  Here it is the sun and view zenith angles; in Case 3 it is the time step.
  The rule should say derived numeric geometry covers space, time and viewing geometry.
- **Gap: the replaced stage was fitted.**
  The packaged matched filter fits μ and Σ per scene; the new kernel fits them inside.
  `replace` must accept either a fitted operator or a stateless one, and say which the slot expects.

## Case 6: AEMET station series, several variables

**Who and goal.**
A climate analyst validates ERA5 2 m temperature and 10 m wind against 40 AEMET stations around Valencia for 2025: hourly station records in, per-station bias and RMSE out.

**Where it sits.**
Points with a time dim and a variable dim, eager, built-in operators, a short custom graph.
Two equally valid ways exist to sample ERA5 at the stations, one per run mode.

```python
# (8760, 2, 40): hourly, t2m and ws10, 40 stations; elevation in the z leaf; missing reports = fill
stations: GeoPoints["time variable obs"] = xrproducts.aemet.stations(
    region=valencia, year=2025, vars=["t2m", "ws10"]
)
# ERA5 on (time, lat, lon), lazy
era5: Dataset["time lat lon"] = xrproducts.cds.open(
    "era5_single_levels", vars=["t2m", "ws10"], region=valencia, year=2025
)
# the two variables stacked on one dim
era5_v: DataArray["variable time lat lon"] = era5.to_array("variable")

# way 1, lazy, by value: native interpolation onto the station coordinates
lon_s: DataArray["obs"] = stations.lon_da()
lat_s: DataArray["obs"] = stations.lat_da()
model_a: DataArray["time variable obs"] = era5_v.interp(lon=lon_s, lat=lat_s)

# way 2, eager, per item: a named sampler mapped over time
# both use the same bilinear weights and edge rule, held by a contract test ← CS-F10
grid_series: GeoArray["variable time y x"] = GeoArray.from_xarray(era5_v)
sample: Operator = geoarray.SampleAt(stations)
model_b: GeoPoints["time variable obs"] = geoarray.map_over(
    sample, grid_series, dim="time"
)

# m − y per hour, station and variable
error: GeoPoints["time variable obs"] = model_b - stations
# bias(v, s) = (1/T) ∑ₜ [m(t, v, s) − y(t, v, s)], over valid t only
bias: GeoPoints["variable obs"] = geoarray.reduce(error, np.nanmean, dim="time")
# for the analyst: one row per (station, variable) ← CS-F10
table: pd.DataFrame = bias.to_dataframe()
```

**What it stresses.**

- **Gap: two samplers must agree.**
  `xr.interp` and `geoarray.sample` both put a grid onto points.
  If their bilinear weights or edge handling differ, the two ways give different biases.
  A contract test should hold them equal, or one should be defined as the other.
- **Gap: analysts want tables.**
  Station data lives naturally in a DataFrame; `GeoTimeSeries` needs `to_dataframe()` / `from_dataframe()` the way grids have `to_xarray()`.
- **Holds:** variables are a dim, time is a dim for fixed stations, and fill values are masked; the types from the geoarray page cover this without a new carrier.
- **Small:** station elevation belongs in the `z` leaf, so a lapse-rate correction can read it as derived geometry.

## Case 7: an operational daily alert pipeline

**Who and goal.**
An operations team runs a scheduled job every morning: yesterday's TROPOMI XCH₄ over a watchlist of 300 facilities in, hotspot alerts out, with no one writing code day to day.

**Where it sits.**
The far top right: a packaged pipeline, packaged runner, built-in operators only.
Eager throughout; the runner fans out one task per orbit.

What `tropomi_hotspots.v3` does inside:

```python
def tropomi_hotspots(
    date: str, watchlist: str, runner: Runner, store: RunStore
) -> Detections:
    # about 14 orbits a day
    orbits: CatalogRows = geocatalog.query("tropomi-l2-ch4", date=date)
    # concurrent reads, qa ≥ 0.5 ← CS-F4
    read: Operator = geoproducts.tropomi.ReadL2(qa_min=0.5)
    swaths: list[GeoSwath["row col"]] = runner.map(read, orbits)
    # 300 facilities
    sites: GeoPoints["obs"] = GeoPoints.from_parquet(watchlist)

    # X̄ within 10 km; X̄_bg and σ_bg over a 30–80 km annulus, per site
    disc_stats: Operator = geoops.points.DiscAnnulusStats(
        inner_km=10, outer_km=(30, 80)
    )
    x: GeoPoints["obs"]
    x_bg: GeoPoints["obs"]
    s_bg: GeoPoints["obs"]
    x, x_bg, s_bg = disc_stats(swaths, sites)
    # ΔX = X̄ − X̄_bg ; z = ΔX / σ_bg
    enhancement: GeoPoints["obs"] = x - x_bg
    z: GeoPoints["obs"] = enhancement / s_bg

    # state across runs, keyed by (pipeline version, date) ← CS-F4, CS-F11
    yesterday: str = prev_day(date)
    z_prev: GeoPoints["obs"] = store.read("z", date=yesterday)
    # idempotent: same key, same value ← CS-F11
    store.write("z", z, date=date)

    # alert when z > 3 on two consecutive days
    hit: GeoPoints["obs"] = (z > 3) & (z_prev > 3)
    return Detections.from_points(
        sites[hit], enhancement=enhancement[hit], sigma=s_bg[hit], z=z[hit]
    )
```

What the scheduler calls:

```python
# 16 concurrent workers
runner: Runner = pipekit.runners.Concurrent(max_workers=16)
# the versioned pipeline, run for one date ← CS-F11
pipeline: Pipeline = pipelines.tropomi_hotspots.v3
run: PipelineRun = pipeline.run(
    date="2026-10-21", watchlist="facilities_2026q4.parquet", runner=runner
)

# GeoPoints["variable obs"]: enhancement, σ, z ← CS-F6
alerts: Detections = run.output
# pipeline 3.x, operator versions, input checksums ← CS-F11
prov: Provenance = run.provenance
```

**What it stresses.**

- **Gap: pipelines need versions and provenance.**
  An alert must say which pipeline version and which inputs produced it, and re-running a date must give the same alert.
  The vision says nothing about versioning or idempotent runs.
- **Gap: state across runs.** "Two consecutive days" means today's run reads yesterday's output: the same stateful-runner need as Case 3, on a daily cycle.
- **Holds:** nobody in this case touches a carrier; the readable output type (CS-F6) is what they consume.

## Case 8: a new finite-volume scheme, pure L1

**Who and goal.**
A numerics developer adds a WENO5 advection scheme to finitevolX and tests it on a rotating-cone problem: no satellite data, no files, only arrays and a grid.

**Where it sits.**
The far bottom left: their own kernel, their own `lax.scan` loop, compiled.
GeoStack is touched once, if at all, to build a grid bundle and to wrap the final field for a plot.

```python
# carrier-free derived geometry: 512 × 512 at 1 km ← CS-F12
grid: CGrid = finitevolx.grids.uniform(nx=512, ny=512, dx=1_000.0, dy=1_000.0)
# initial tracer and a solid-body rotation
q0: Float[Array, "H W"] = rotating_cone(grid)
u: Float[Array, "H W"]
v: Float[Array, "H W"]
u, v = solid_body_rotation(grid)


# L1, in finitevolX: n steps of WENO5 advection, arrays and a grid bundle only
def weno5_advect(
    q0: Float[Array, "H W"],  # tracer
    u: Float[Array, "H W"],  # zonal velocity, m s⁻¹
    v: Float[Array, "H W"],  # meridional velocity, m s⁻¹
    grid: CGrid,  # spacings and areas (derived geometry)
    n_steps: int,
) -> Float[Array, "H W"]:

    def tendency(q: Float[Array, "H W"]) -> Float[Array, "H W"]:
        # WENO5 face fluxes (Fₓ, Fᵧ)
        fx: Float[Array, "H W"]
        fy: Float[Array, "H W"]
        fx, fy = weno5_flux(q, u, v, grid)
        # ∂q/∂t = −∇·(𝐮 q)
        return -divergence(fx, fy, grid)

    def step(q: Float[Array, "H W"], _: None) -> tuple[Float[Array, "H W"], None]:
        # RK3 in time with step Δt
        q_next: Float[Array, "H W"] = rk3(q, tendency, DT)
        return q_next, None

    q_final: Float[Array, "H W"]
    q_final, _ = jax.lax.scan(step, q0, None, length=n_steps)
    return q_final


# the developer's own test: pure L1, no carrier anywhere
q_final: Float[Array, "H W"] = weno5_advect(q0, u, v, grid, n_steps=2_000)
```

The way up, once the scheme works.
Someone else wants to advect a real SST anomaly map, georeferenced, without knowing about `CGrid`.
A `KernelOp` gives them an operator on carriers without anyone writing a wrapper class: it maps the carrier's `(y, x)` dims onto the kernel's `(H, W)` axes, builds `grid` from the carrier's transform, calls the kernel and rewraps the result with the same georeference.

```python
# L2, in somax[geo]: GeoModels may import finitevolX and GeoStack; finitevolX may import neither ← CS-F12
Advect: type[KernelOp] = KernelOp.from_function(
    weno5_advect,
    "(h,w),(h,w),(h,w)->(h,w)",  # q, u, v in; q out; h, w consumed
    core={"h": "y", "w": "x"},  # which carrier dims are H and W
    geometry={
        "grid": somax.geo.cgrid_from_carrier
    },  # transform + CRS → CGrid, per call
)
advect: KernelOp = Advect(n_steps=2_000)

# eager: one georeferenced map in, one out, same grid and CRS
sst_later: GeoArray["y x"] = advect(sst_anom, u_map, v_map)

# lazy: a daily cube, the kernel run once per day and per chunk
sst_cube_later: DataArray["time y x"] = advect(sst_cube, u_cube, v_cube)
```

**What it stresses.**

- **Holds:** the vision must leave this user alone.
  Nothing here needs a carrier, an operator or a runner.
- **Gap: derived-geometry helpers need a carrier-free path.**
  `uniform(...)` builds a `CGrid` from numbers; `cgrid_from_transform` builds one from a carrier.
  Both must exist: the first in finitevolX, importing nothing from GeoStack; the second in the package above both (somax[geo]), since finitevolX may not import a carrier.
- **Gap: the way up.**
  Once WENO5 works, wrapping it as an L2 operator should be one line (`KernelOp` or a thin `Operator`), and the docs should show that path.

## Case 9: GeoBench, a packaged benchmark running submitted models

**Who and goal.**
A researcher submits a new SSH-mapping model to GeoBench: the benchmark owns the data, splits, metrics and runner; the submission owns only the model.

**Where it sits.**
Top left: packaged orchestration around user computation.
The benchmark is the packaged pipeline; the model is whatever the user wrote, down to raw L1.

What `ssh_mapping_osse.v1.evaluate` does inside:

```python
def evaluate(model: SSHMapper, runner: Runner) -> Scores:
    # splits fixed by the task from its own seed; truth held by the runner ← CS-F13
    train: Split
    test: Split
    train, test = task.splits()
    fitted: SSHMapper = model.fit(train)

    scores: list[Score] = []
    for window in runner.iterate(test):
        # the model sees only the observations (GeoTracks)
        obs: GeoPoints["obs"] = window.obs
        pred: GeoArray["time y x"] = fitted.predict(obs, task.grid)
        # contract check before scoring: dims, grid and time stamps ← CS-F1
        geoarray.types.check(pred, GeoImageSeries, grid=task.grid, time=window.times)
        # RMSE = √(𝔼[(η̂ − η)²]); λₓ = wavelength where the PSD score crosses 0.5
        score: Score = task.metrics(pred, window.truth)
        scores.append(score)

    return task.aggregate(scores, by=("region", "season"))
```

What the submission writes:

```python
# the contract, in readable types:
# GeoTracks ≡ GeoPoints["obs"] + a time leaf; GeoMap ≡ GeoArray["y x"]; GeoImageSeries ≡ GeoArray["time y x"]
class SSHMapper(Protocol):
    def fit(self, train: Sequence[tuple[GeoTracks, GeoImageSeries]]) -> "SSHMapper": ...
    def predict(self, obs: GeoTracks, grid: GeoMap) -> GeoImageSeries: ...


class MyMapper:
    def predict(self, obs: GeoTracks, grid: GeoMap) -> GeoImageSeries:
        # crossing with padding ← CS-F3
        jobs: GeoPoints["obs"] = obs.to_jax(pad_to=20_000)
        times: Int[Array, "T"] = obs.window.times
        # bin observations per day onto the grid, valid points only
        x: Float[Array, "T H W"] = bin_tracks(jobs, grid, times=times)
        # η̂ = f_θ(x)
        y: Float[Array, "T H W"] = self.net(x)
        return GeoArray.from_frames(y, like=grid, time=times)


# score the submission with the task's own runner
runner: Runner = pipekit.runners.Local()
task: Task = geobench.tasks.ssh_mapping_osse.v1
scores: Scores = task.evaluate(MyMapper(), runner=runner)
```

**What it stresses.**

- **Holds, and is the strongest use of readable types:** the submission contract is written in them.
  `GeoTracks` in, `GeoImageSeries` out is a precise, checkable interface that a shape-only contract cannot express.
- **Gap: the contract needs checking, not just annotating.**
  The benchmark must refuse a submission whose output has the wrong dims, grid or time stamps before scoring, which is CS-F1's "type as a checkable signature" again.
- **Gap: leakage is the runner's job.**
  The packaged runner, not the submission, must guarantee the model never sees test-time observations; the vision should say runners own splits.

## Case 10: a flaring index as a custom graph, promoted to a pipeline

**Who and goal.**
An analyst builds a gas-flaring detector on Landsat 8/9 SWIR, uses it for a month as a notebook graph, then promotes it to a packaged pipeline the team can run.

**Where it sits.**
The centre: one new kernel wrapped as a `KernelOp`, built-in operators around it, a custom pipekit graph.
Eager, then packaged.

```python
# L1: a thermal-anomaly index on two SWIR bands, arrays only
def flare_index(
    swir1: Float[Array, "..."], swir2: Float[Array, "..."]
) -> Float[Array, "..."]:
    # I = (ρ₇ − ρ₆) / ρ₆, ρₖ the reflectance of Landsat band k (placeholder index)
    return (swir2 - swir1) / swir1


# L2: wrapped once, band names as inputs; GeoArray["band y x"] → GeoArray["y x"]
FlareIndex: type[KernelOp] = KernelOp.from_function(
    flare_index,
    "(),()->()",
    inputs={"swir1": "B6", "swir2": "B7"},
    dim="band",
)

# the stages, each named
read: Operator = geoproducts.landsat.Read()  # → GeoArray["band y x"]
mask: Operator = geoops.CloudMask(apply=True)  # → GeoArray["band y x"], clouds → fill
index: Operator = FlareIndex()  # → GeoArray["y x"]
threshold: Operator = geoops.Threshold(0.15)  # → GeoArray["y x"], bool
vectorize: Operator = geoops.geom.Vectorize()  # → Detections; eager only ← CS-F8

graph: pipekit.Graph = pipekit.Graph(
    read=read, mask=mask, index=index, threshold=threshold, vectorize=vectorize
)
flares: Detections = graph.run(scene_ids)

# a month later: versioned, typed, registered ← CS-F14
pipelines.register(
    "landsat_flares", graph, version="1.0", inputs=GeoImage, outputs=Detections
)
```

The index formula is a placeholder for whichever SWIR thermal-anomaly index the analyst picks.

**What it stresses.**

- **Gap: the promotion path.**
  The vision has a way down (pull a stage out) but no way up: a graph that works should become a packaged, versioned pipeline in one call, with its readable input and output types declared then.
- **Holds:** the new kernel is ten lines and runs eagerly here, under `jit` in Case 1's training loop, and lazily on a cube, with no further code.
- **Small:** `KernelOp.from_function` with band names as inputs avoids the hand-written `NDVI`-style operator class; worth making the default.

## Case 11: a GP interpolation of SST with your own fitting loop

**Who and goal.**
A statistician maps daily SST in the Mediterranean from drifter and Argo surface points with a Gaussian process, fitting the kernel hyperparameters themselves, and wants the predictive variance as well as the mean.

**Where it sits.**
Bottom left: their own model (kernellib / pyrox at L1), their own optax loop, compiled.
Readable types appear only at the two ends.

```python
# in: surface points for one day
raw: GeoPoints["obs"] = xrproducts.cmems.insitu("sst", region=med, date="2026-08-15")
# crossing with padding to N = 4 096 ← CS-F3
obs: GeoPoints["obs"] = raw.to_jax(pad_to=4_096)
# out: a 0.05° grid over the Mediterranean
grid: GeoArray["y x"] = geoarray.grid(bounds=med, resolution=0.05, crs="EPSG:4326")

# derived geometry: lon/lat → unit vectors on the sphere, so distances are chordal ← CS-F5
xo: Float[Array, "N 3"] = geoarray.to_unit_sphere(obs)
xg: Float[Array, "M 3"] = geoarray.to_unit_sphere(grid)  # M = H · W


def nll(theta: GPParams) -> Float[Array, ""]:
    # K = k_θ(X, X) + σ²I on valid points; X the N observation locations
    # −log p(y | θ) = ½ yᵀK⁻¹y + ½ log|K| + (N/2) log 2π
    return gp_nll(theta, xo, obs.data, mask=obs.valid)


# your own loop: 500 Adam steps
optimizer: optax.GradientTransformation = optax.adam(1e-2)
theta: GPParams = fit(nll, theta0, optimizer, steps=500)

# μ* = k*ᵀK⁻¹y ; σ*² = k** − k*ᵀK⁻¹k*, k* = k_θ(X, x*)
mean: Float[Array, "M"]
var: Float[Array, "M"]
mean, var = gp_predict(theta, xo, obs.data, xg, mask=obs.valid)

# the standard summary dim, labels ("mean", "var") ← CS-F15
sst: GeoArray["stat y x"] = grid.with_stats(mean=mean, var=var)
```

**What it stresses.**

- **Gap: uncertainty has no convention.**
  Mean and variance here, members in Case 12, quantiles elsewhere.
  The dim conventions (`time`, `member`, `level`, `band`, `variable`, `patch`) need one for distribution summaries, e.g. `moment` or `stat`, with standard labels.
- **Holds:** derived geometry (unit vectors on the sphere) comes from a helper, and the model never sees a CRS.
- **Holds:** padding with `valid` (CS-F3) is exactly what a GP needs for a fixed-size kernel matrix.

## Case 12: an ensemble forecast with a packaged runner over members

**Who and goal.**
A modeller runs a 32-member shallow-water ensemble from perturbed initial SSH for a 10-day forecast, and wants the ensemble written as a cube without writing the loop.

**Where it sits.**
Top centre: a packaged runner over a built-in model (somax), compiled inside, lazy on output.

What the runner does inside:

```python
# pipekit.runners.Ensemble — sketch of its internals
def run_ensemble(
    model: Algorithm,
    x0: GeoArray["y x"],
    perturb: Perturbation,
    members: int,
    chunk: int,
    times: Int[Array, "T"],
    seed: int,
    out: Zarr,
) -> None:
    # one seed per run ← CS-F16
    key: PRNGKeyArray = jax.random.key(seed)
    # the algorithm's exposed L1 core, vmapped over members, compiled once ← CS-F16
    batched: Callable = jax.vmap(model.l1, in_axes=(0, None))
    run_chunk: Callable = jax.jit(batched)
    perturb_many: Callable = jax.vmap(perturb, in_axes=(None, 0))

    for start in range(0, members, chunk):
        # one key per member in this chunk
        chunk_key: PRNGKeyArray = jax.random.fold_in(key, start)
        keys: PRNGKeyArray = jax.random.split(chunk_key, chunk)
        # x₀⁽ᵐ⁾ = x₀ + δ⁽ᵐ⁾, δ⁽ᵐ⁾ a random field per member m
        x0s: Float[Array, "M H W"] = perturb_many(x0.data, keys)
        # trajectories: (M, T, H, W)
        traj: Float[Array, "M T H W"] = run_chunk(x0s, times)
        # crossing: rewrap once per chunk, append on the member dim
        block: GeoArray["member time y x"] = GeoArray.from_frames(
            traj, like=x0, dims=("member", "time", "y", "x")
        )
        out.append(block.to_xarray(), dim="member")
```

What the user writes:

```python
# initial state: e.g. Case 3's last analysis
analysis_today: GeoArray["y x"] = ...
# the model, the perturbation and the output
model: Algorithm = somax.ShallowWater(depth=4_000.0, dt=600.0)
perturb: Perturbation = somax.perturb.gaussian_random_field(
    sigma=0.02, length_scale=50e3
)
out: Zarr = Zarr("ssh_ens_20261022.zarr")

# 32 members in chunks of 8, one seed for the run
runner: Runner = pipekit.runners.Ensemble(members=32, chunk=8, seed=20261022)
ens: EnsembleRun = runner.run(
    model, x0=analysis_today, perturb=perturb, times=forecast_times, out=out
)

# the ensemble cube, lazy
cube: DataArray["member time lat lon"] = xr.open_zarr(ens.out).ssh
# η̄ and spread = √(1/(M−1) ∑ₘ (η⁽ᵐ⁾ − η̄)²), on the stat dim ← CS-F15
stats: Operator = xrops.EnsembleStats(stats=("mean", "std"))
summary: DataArray["stat time lat lon"] = stats(cube)
```

**What it stresses.**

- **Gap: models need a named L1 entry point.**
  The runner calls `model.l1_loop` to vmap the raw loop; an algorithm that only exposes its L3 signature (`simulate(GeoMap) -> GeoImageSeries`) cannot be vmapped without rewrapping per member.
  L3 algorithms should expose their L1 core.
- **Gap: seeds belong to the runner.**
  One key per member, derived from the run's seed; the vision should say randomness is passed down, never drawn inside operators.
- **Holds:** the `member` dim, `vmap` over it, and lazy writing on `append_dim` all fit the existing conventions.

## Case 13: a labelling app serving detections on demand

**Who and goal.**
A geoapps backend: a labeller pans to a facility, the app fetches the latest Sentinel-2 scene, runs the plume detector and returns a candidate mask within a few seconds.

**Where it sits.**
Right, upper: built-in operators and a trained model behind a web handler.
Orchestration is the web framework's, not pipekit's.
Eager for I/O, compiled for the model, one scene at a time.

```python
# start-up, once per worker: slim imports (the slim geoops base, decided in the operators doc) and an ahead-of-time compile ← CS-F17
loaded: plumes.Detector = plumes.Detector.load("detector_v7")
detector: JaxOperator = pipekit_jax.to_jax(loaded)
# the one shape the app serves: 13 bands, 512 × 512
spec: GeoArraySpec = GeoArray.spec("band y x", shape=(13, 512, 512))
lowered: jax.stages.Lowered = jax.jit(detector).lower(spec)
detect: jax.stages.Compiled = lowered.compile()
vectorize: Operator = geoops.geom.Vectorize()  # eager-only geometry operator ← CS-F8


@app.get("/detect")
async def detect_at(lon: float, lat: float) -> GeoJSON:
    # async only at the I/O edge
    scene: GeoArray["band y x"] = await geoproducts.sentinel2.read_latest_async(
        lon, lat, size=512
    )
    # crossing with padding to the compiled shape ← CS-F3
    jscene: GeoArray["band y x"] = scene.to_jax(pad_to={"band": 13, "y": 512, "x": 512})
    # p(y, x) ∈ [0, 1], the plume probability per pixel
    prob_jax: GeoArray["y x"] = detect(jscene)
    prob: GeoArray["y x"] = prob_jax.to_numpy()
    mask: GeoArray["y x"] = prob > 0.5
    # mask → polygons
    found: Detections = vectorize(mask)
    return found.to_geojson()
```

**What it stresses.**

- **Gap: latency budgets.**
  Import time (xarray alone was about 0.57 s in one exploratory run) and the first `jit` compile land on the first request.
  The app needs ahead-of-time compilation and a slim import path: the slim geoops base (X1) matters here more than anywhere.
- **Holds:** async stays at the I/O edge, as the vision says; the operators are plain functions the handler calls.
- **Gap: padding also for images.**
  `pad_to` on a raster, so every scene fits the one compiled shape, generalises CS-F3 beyond points.

## What the cases change in the vision

Three findings change a principle (CS-F1, CS-F2, CS-F11); the rest add requirements, most of them to the pipelines page that does not exist yet.

**What held in every case:** one named crossing per mode change; I/O never under JAX; one L1 kernel serving every carrier; a stage swapped without touching the others; generalized-ufunc specs expressing per-column and per-pixel work without loops.

| ID | Finding | Cases | Proposed edit |
| --- | --- | --- | --- |
| CS-F1 | Readable types stop at `GeoArray`, so lazy inputs have none, and contracts cannot be checked on them | 4, 9 | **Principle 5 changes:** a readable type is a dim signature (`GeoImageSeries` = `time y x` on a grid) that any registered carrier can match, checked with `Signature.matches`; "one object" becomes "one type name on any carrier" |
| CS-F2 | One L3 algorithm takes carriers from two families | 2, 3 | **Principle 4 gains a clause:** inside an L3 algorithm, several families may meet; the algorithm does its own crossings, by value in xarray first |
| CS-F3 | Fixed shapes before `jit`, for points and images | 3, 11, 13 | padding is part of the named crossing: `to_jax(pad_to=...)` sets the `valid` leaf |
| CS-F4 | The runner has jobs: fan out I/O, carry state between cycles and runs | 1, 3, 7 | the pipelines page specifies concurrent I/O stages with a join, and stateful cycling with warm start and append-to-Zarr |
| CS-F5 | Derived geometry is more than spacing | 3, 5, 11 | the primitives rule says space, time and viewing geometry (Δx, Δt, angles, unit vectors on the sphere) |
| CS-F6 | Results can be records, not rasters | 2, 7, 10 | a readable type for detections and estimates on `GeoPoints["variable obs"]`, provisionally `Detections` |
| CS-F7 | Native xarray operations are not pipeline stages | 4 | xrops wraps the few that pipelines need (`Climatology`, `Anomaly`, `Resample`) as named operators |
| CS-F8 | Some operators can never compile (GDAL rasterize, vectorize) | 1, 10, 13 | operators declare `backends`; the docs list eager-only ones (operators page, [OP-O6](./operators.md)) |
| CS-F9 | A replaced stage may have been fitted | 5 | each pipeline slot says whether it expects a fitted or a stateless operator; `replace` checks |
| CS-F10 | Two ways to sample a grid at points; analysts want tables | 6 | a contract test holds `geoarray.sample` equal to `xr.interp`; `GeoPoints.to_dataframe()` / `from_dataframe()` |
| CS-F11 | Production runs need versions, provenance and idempotence | 7 | **a new principle:** a packaged pipeline is versioned; every output records pipeline version, operator versions and input checksums; re-running a key gives the same output |
| CS-F12 | Pure-L1 users must be left alone, and given a way up | 8 | geometry helpers have carrier-free constructors; the docs show L1 → `KernelOp` → pipeline as one path |
| CS-F13 | Splits and leakage belong to the runner | 9 | runners own splits and hold back truth; a submission or stage never sees test data |
| CS-F14 | No way up from a working graph | 10 | `pipelines.register(name, graph, version, inputs, outputs)` promotes a graph, declaring its readable types |
| CS-F15 | Uncertainty has no dim convention | 11, 12 | add a dim convention for distribution summaries (`moment` or `stat`, labels `mean`, `var`, `q05`…) beside `member` |
| CS-F16 | Runners need the L1 core of an algorithm, and own randomness | 12 | every L3 algorithm exposes its L1 loop; seeds are passed down from the run, never drawn inside operators |
| CS-F17 | Latency: imports and first compile | 13 | ahead-of-time compile for fixed shapes; the slim geoops base (decided on the [operators page](./operators.md)) is required, not optional, for apps |

Three to decide first: CS-F1 changes what a readable type is; CS-F2 and CS-F11 change principles.
CS-F4, CS-F11, CS-F13, CS-F14 and CS-F16 together are most of the missing pipelines page.
