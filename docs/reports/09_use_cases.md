---
title: Use cases across the stack
subtitle: Report 9 · Twenty deployment shapes mapped onto the libraries as they ship
short_title: 9 · Use cases
description: Twenty use cases, from a notebook to a regulatory artifact to a validation queue, each mapped onto pipekit, its sister packages, geocatalog, geopatcher, geotoolz, xrtoolz and georeader, with every call marked as shipped today or proposed.
keywords:
  - use cases
  - deployment
  - pipekit
  - geocatalog
  - geopatcher
  - geotoolz
  - geoapps
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

The earlier reports describe each library on its own, whereas this one tests them together against the shapes that real work takes: a notebook, an ETL job, a training loop, an inference API, a tile server, a regulatory artifact, a validation queue and a scheduled reanalysis. Thirteen cases come from the original gallery, which we rewrote in the names that ship. Seven are new: six of them come from the geoapps design (validation, scheduled pipelines, what-if reruns, catalog exploration, the reinforcement loop and event kinds) and one comes from the xarray side. We mark every call as shipped today or as a proposal, and the proposals (lettered A to I) are the stack's backlog.

+++

:::{seealso} The domain-neutral version lives in pipekit
pipekit's own docs distil these cases into fourteen [deployment patterns](https://jejjohnson.github.io/pipekit/patterns/), each with its pieces, its status and a runnable tutorial, but without the geo libraries. This report stays the stack-wide view, so it keeps the methane plume, the libraries each case needs and the proposals A to I.
:::

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Use-case mapping; supersedes Report 4 |
| **Reading time** | about 40 minutes |
| **Audience** | anyone pressure-testing the library structure against real deployments, or deciding what to build next |
| **Companions** | [Report 2](02_pipekit_surface.md) (pipekit core), [Report 3](03_sister_libraries.md) (sister libraries), [Report 6](06_geocatalog.md) (geocatalog), [Report 7](07_geopatcher.md) (geopatcher), [Report 16](16_geo_data_stack.md) (the geo data stack); [the GeoModels package boundaries](../geomodels/package_boundaries.md), [radtrax · 1](../geomodels/radtrax/01_column_retrieval.md) and [plumax · 1](../geomodels/plumax/01_emission_rate.md) for the methane physics |
| **Checked against** | the revisions pinned in `references/`: pipekit and its six sister packages at 0.0.2 (`ef3c17f`); geotoolz 0.10.2 with geocatalog 0.5.1 and geopatcher 0.10.2 (`088475e`); xrtoolz 0.0.5 (`308c7f2`); georeader 2.4.1. The running example's Graph, the registry, the YAML loader and the wind-swap gate are from real runs |
| **Reference** | the original gallery, [`usecases.md`](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/plans/geotoolz/examples/usecases.md); the geoapps design plan for cases 14 to 19 |
:::

:::{important} How to read the code
Every code block follows the stack's sketch conventions:
- a **shapes header** names each dimension;
- a **unicode equation** sits above each operation;
- the **shape transition** runs down the right margin.

Every call into the stack carries a [status marker](../overview_examples.md#ex-markers) (`today`, `user` or `proposed (X)`), where X is a proposal letter in [](#r9-proposals), and `# arguments sketched` flags the arguments that we did not run.
:::

(r9-shape)=
## The library shape, as it ships

```{figure} ../images/reports/r9_stack.svg
:label: fig-r9-stack
:alt: The stack in four layers. User code calls geotoolz, xrtoolz, geopatcher and geocatalog side by side. geotoolz and xrtoolz depend on pipekit; geopatcher reaches pipekit only through its pipekit extra; geocatalog has no pipekit dependency. Inside the pipekit box are its sister packages: array, train, experiment, evaluate, cycle and jax. georeader, geocloud and xarray form the substrate at the bottom.
:width: 100%

**The stack the cases run on, with only the dependencies that exist.** Solid arrows are runtime dependencies; the dashed one is an optional extra.
```

[](#fig-r9-stack) is the stack the cases run on, and [Report 16](16_geo_data_stack.md#r16-ownership) sets out its layering and ownership rules. One consequence of those rules shapes every case below. geocatalog and geopatcher are used on their own, next to the domain libraries, and the deployment infrastructure (YAML, hashing, the registry, artifacts) sits in pipekit-experiment rather than in pipekit core.

(r9-running)=
## The running example: one methane plume

Every case processes the same kind of data, so their code lands on the same arrays.

**The scene.** An imaging spectrometer such as EMIT measures a radiance cube $\mat{Y} \in \mathbb{R}^{B \times H \times W}$, which has $B$ spectral bands over an $H \times W$ pixel grid with a pixel area of $\Delta A$ (60 m pixels give $\Delta A = 3600$ m²). We index the pixels by $n = 1, \ldots, HW$, and $\vect{y}_n \in \mathbb{R}^{B}$ is the spectrum of pixel $n$. A methane plume absorbs in the shortwave infrared, so it shows up as a small, spectrally specific dip in $\vect{y}_n$.

**The enhancement.** A matched filter estimates the column enhancement $\hat{\alpha}_n \in \mathbb{R}$ at each pixel from the background mean $\gvect{\mu}_y \in \mathbb{R}^{B}$ and covariance $\gmat{\Sigma}_y \in \mathbb{R}^{B \times B}$ of the spectra and the methane target signature $\vect{J}_h \in \mathbb{R}^{B}$. Its derivation and its limits are in [radtrax · 1, stage 5](../geomodels/radtrax/01_column_retrieval.md#rx1-stage-5).

**The flux.** Converting $\hat{\alpha}_n$ to a mass column $\Omega_n \in \mathbb{R}$ and masking the plume pixels $\mathcal{P} \subset \{1, \ldots, HW\}$ gives the integrated mass enhancement $\mathrm{IME} = \sum_{n \in \mathcal{P}} \Omega_n \Delta A$, and then the source rate $Q = U_\text{eff} \mathrm{IME} / \ell$ for an effective wind speed $U_\text{eff}$ and a plume length $\ell$ [@varon2018]. The derivation, and which length goes with which wind calibration, are in [plumax · 1](../geomodels/plumax/01_emission_rate.md#er1-ime). Which package owns each step once radtrax and plumax exist is set by the [GeoModels package boundaries](../geomodels/package_boundaries.md#pb-moves). On this page, $\Omega$ is the methane mass column and not a spatial domain, and $\mathrm{IME}$, $Q$, $U_\text{eff}$ and $\ell$ are all scalars.

Every piece of this ships in geotoolz, and pipekit's `Graph` composes them, because `IMEEstimate` takes two inputs (the mass column and the mask):

```{code-block} python
:caption: The running example as a pipekit Graph, from a real run on a synthetic 60-band scene with a 1,500 ppm·m plume.
:linenos:

# Shapes: B = 60 bands; H = W = 64 pixels; ΔA = 3600 m²
import pipekit as pk
import geotoolz as gz

radiance = pk.Input("radiance")                          # Y: (B, H, W)          today
# α_n = J_hᵀ Σ_y⁻¹ (y_n − μ_y) / (J_hᵀ Σ_y⁻¹ J_h)
enh = gz.matched_filter.MatchedFilter(
    target=j_h, fit_on_call=True)(radiance)              # (B, H, W) → (H, W)   today
# Ω_n = α_n · ρ_CH₄    (ppm·m → kg m⁻²)
mass = gz.plume.ColumnToMass(gas="CH4", units_in="ppm_m",
                             units_out="kg_m2")(enh)     # (H, W) → (H, W)      today
# 𝒫 = {n : α_n > τ_Otsu}, connected components ≥ 20 px
mask = gz.plume.PlumeMask(threshold="otsu", min_area_px=20)(enh)  # bool (H, W) today
# IME = Σ_{n∈𝒫} Ω_n ΔA;   Q = U_eff · IME / ℓ
flux = gz.plume.IMEEstimate(wind_speed=3.0)(mass, mask)  # (H, W)² → dict       today

plume = pk.Graph(inputs={"radiance": radiance},
                 outputs={"enh": enh, "mask": mask, "flux": flux})   # today
out = plume(radiance=scene)
out["enh"].values.max()       # → 1518.8 ppm·m        (1500 injected)
out["flux"]
# → {'ime_kg': 595.1, 'length_m': 1994.5, 'wind_speed_m_s': 3.0,
#    'emission_rate_kg_s': 0.895, 'emission_rate_uncertainty_kg_s': 0.448}
```

`plume` (radiance in; enhancement, mask and flux out) is the object that most cases below pass around. Cases that need a single-input, single-output pipeline use a `Sequential` of the same operators instead.

(r9-proposals)=
## What the cases need that does not ship yet

Running the original gallery against the pinned code leaves nine gaps ([](#tbl-r9-proposals)). Each gap gets a letter, and every `proposed (X)` marker in the cases points at one of them:

:::{table} The proposals the cases depend on, in the move schema: symbol, destination, reason.
:label: tbl-r9-proposals

| ID | Symbol | Destination | Reason, and what exists today |
|---|---|---|---|
| **A** | `to_yaml` that emits nested `_target_` configs | pipekit-experiment `adapters.hydra` | a hand-written nested YAML loads today; dumping a built nested pipeline emits pipekit's debug envelope, which does not load back |
| **B** | `pipeline_hash(op)` | pipekit-experiment | the registry already computes a stable content hash on `store`; there is no public way to hash without storing |
| **C** | `CatalogPipeline`, `CatalogReader` | `geocatalog.integrations.pipekit` | assumed by Reports [2](02_pipekit_surface.md#r2-group-j), [3](03_sister_libraries.md#r3-todos) and 6 and by cases 2, 10, 12 and 15; not built ([Report 6](06_geocatalog.md#r6-open-questions), question 4) |
| **D** | `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction` | pipekit-array phase C | promised there; not shipped anywhere |
| **E** | `AssertCRSEquals`, `AssertResolutionWithin`, `AssertSchema` | `geotoolz.qa` (or a new `geotoolz.checks`) | `geotoolz.qa` holds QA-bit masks today, not assertions |
| **F** | an allowlisted YAML loader | pipekit-experiment, beside `HydraConfigLoader` | `pipekit.loads_sandboxed` allowlists flat JSON only |
| **G** | `config_diff`, an A/B harness | pipekit-evaluate | `Benchmark` scores estimators on a task; it does not diff two pipelines |
| **H** | `ActiveSampler` | `geotoolz.patch_ops` | the patcher and scorers exist; the selection loop does not |
| **I** | SAR and fusion operators (`LinearToDB`, `AlignAndStack`) | geotoolz | despeckling ships (`restore.DespeckleLee`); dB conversion and grid alignment do not |
:::

(r9-explore)=
## Exploration and batch processing

(r9-case-1)=
### Case 1 · Notebook exploration (pipekit → geotoolz)

**TL;DR.** Look at a scene through a cleaned, clipped index, and print what each step does to it.

**Problem.** A Sentinel-2 scene $\mat{X} \in \mathbb{R}^{B \times H \times W}$ with a QA60 band. We want to mask the clouds, stretch each band, compute the NDVI and look at the map.

```{code-block} python
:caption: Case 1. Every call ships today.
:linenos:

# Shapes: B bands (QA60 last), H × W pixels
import numpy as np
import pipekit as pk
import geotoolz as gz

viz = pk.Sequential([
    # x̃ = NaN where QA60 bits 10, 11 are set
    gz.mask.ApplyMask(mask=gz.qa.MaskClouds(qa_band="QA60", bits=[10, 11])),
                                                    # (B, H, W) → (B, H, W)  today
    pk.Tap(lambda gt: print(f"NaN: {np.isnan(gt.values).mean():.1%}")),  # today
    # x̂_b = (x̃_b − p₂)/(p₉₈ − p₂), clipped to [0, 1]
    gz.radiometry.PercentileClip(lower=2.0, upper=98.0),  # (B, H, W)   today
    # NDVI = (ρ_nir − ρ_red)/(ρ_nir + ρ_red + tol)
    gz.indices.NDVI(nir=7, red=3),                  # (B, H, W) → (1, H, W)  today
])
ndvi = viz(scene)                                   # GeoTensor          today
```

**Library map.** pipekit and geotoolz, and the scene came from an earlier georeader call. **Notes.** The case is unchanged from the original gallery apart from the names: masks live in `geotoolz.qa` and are applied with `geotoolz.mask.ApplyMask`, and `PercentileClip` takes `lower` and `upper`.

(r9-case-2)=
### Case 2 · ETL pipelines (geocatalog → pipekit → geotoolz → georeader)

**TL;DR.** Run one saved pipeline over every scene in a catalog, in parallel, and write a COG per scene.

**Problem.** A catalog of $N_\text{scene}$ scenes and a pipeline $f$ defined in YAML. We compute $f(\mat{X}_m)$ for $m = 1, \ldots, N_\text{scene}$ and write each result, while tolerating individual failures.

```{code-block} python
:caption: Case 2. The loader is real today; the driver is proposal C.
:linenos:

# Shapes: N_scene scenes; each X_m: (B, H, W)
from geocatalog import open_catalog
from geocatalog.load import load_raster
from pipekit import ProcessMap
from pipekit_experiment.adapters.hydra import HydraConfigLoader

catalog = open_catalog("s3://bucket/s2_eu.parquet")      # N_scene rows, lazy today
etl = HydraConfigLoader.from_yaml(open("ndvi_etl.yaml").read())  # nested     today

# f(X_m) for m = 1 … N_scene, on 8 processes; failures logged, not raised
CatalogPipeline(                                         # proposed (C)
    catalog=catalog,
    op=etl,
    loader=load_raster,                          # (catalog, slice) → GeoTensor  today
    parallel=ProcessMap(etl, n_workers=8, on_error="log_and_continue"),  # today
).run()                                                  # arguments sketched
```

**Library map.** geocatalog (iteration and loading), pipekit (parallelism), pipekit-experiment (the YAML loader) and geotoolz (the operators named in the YAML, ending in `geotoolz.io.WriteCOG`), with georeader underneath the loader. **Notes.** The YAML is the source of truth. A hand-written nested YAML with `_target_` keys loads today, but dumping a built nested pipeline does not round-trip (proposal A). Every step must pickle for `ProcessMap`, and `pipekit.check_pickleable(etl)` lists the operators that will not ([Report 2](02_pipekit_surface.md#r2-pickle)).

(r9-case-8)=
### Case 8 · Data validation as operators (pipekit → pipekit-array → geotoolz)

**TL;DR.** Put permanent runtime checks into the pipeline, so that CI and production both fail loudly on bad data.

**Problem.** Before and after a correction step, we check the carrier's shape, CRS, resolution, band schema, valid fraction and value range.

```{code-block} python
:caption: Case 8. The carrier-agnostic tier ships; the array and geo tiers are proposals D and E.
:linenos:

# Shapes: X: (B, H, W)
import pipekit as pk
import geotoolz as gz

qc = pk.Sequential([
    pk.AssertShape((5, None, None)),                    # B = 5, any H, W    today
    pk.AssertHasAttribute("crs"),                       # today
    AssertCRSEquals("EPSG:32630"),                      # proposed (E)
    AssertResolutionWithin((9.5, 10.5)),                # proposed (E)
    # valid fraction = |{n : X_n finite}| / (H W) ≥ 0.5
    AssertValidFraction(min_valid=0.5),                 # proposed (D)
    AssertValueRange(min_val=0, max_val=10_000, on_fail="warn"),  # proposed (D)
])
etl = pk.Sequential([
    gz.mask.ApplyMask(mask=gz.qa.MaskClouds(qa_band="QA60", bits=[10, 11])),  # today
    qc,                                                 # raises QCError     today
    gz.radiometry.SimpleAtmosphericCorrection(),        # today; arguments sketched
    qc,
    gz.indices.NDVI(nir=3, red=2),                      # today
])
```

:::{table} The three QC tiers.
:label: tbl-r9-qc

| Tier | Lives in | Names | Status |
|---|---|---|---|
| Carrier-agnostic | `pipekit` | `AssertShape`, `AssertDType`, `AssertHasAttribute`, `AssertCallable`, `Quarantine` | today |
| Array reductions | `pipekit_array` | `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction` | proposed (D) |
| Geo-specific | geotoolz | `AssertCRSEquals`, `AssertResolutionWithin`, `AssertSchema` | proposed (E) |
:::

**Notes.** The split into three tiers ([](#tbl-r9-qc)) is still the right answer to "where does a check go?", but only the first tier exists. Until D lands, `pk.AssertCallable(lambda gt: np.isfinite(gt.values).mean() >= 0.5)` covers the valid fraction, at the cost of a closure (`forbid_in_yaml`). There is no `warn` mode in pipekit's asserts, which raise `QCError`, so `Quarantine` is the non-raising alternative. `TOAToBOA` from the original gallery does not exist, and `SimpleAtmosphericCorrection` is the nearest operator to it.

(r9-case-11)=
### Case 11 · Cross-sensor fusion (the `Graph` case)

**TL;DR.** Combine optical, SAR and a DEM on one grid and classify the stack, keeping the fused stack as a second output.

**Problem.** The inputs are $\mat{X}_\text{opt} \in \mathbb{R}^{B_o \times H \times W}$, $\mat{X}_\text{sar} \in \mathbb{R}^{B_s \times H' \times W'}$ and $\mat{X}_\text{dem} \in \mathbb{R}^{H'' \times W''}$ on three grids. We align them to the optical grid, stack them and predict a class per pixel.

```{code-block} python
:caption: Case 11. The Graph and despeckling ship; dB conversion and alignment are proposal I.
:linenos:

# Shapes: B_o optical, B_s SAR bands; H × W the optical grid; C classes
import pipekit as pk
import geotoolz as gz

opt, sar, dem = pk.Input("optical"), pk.Input("sar"), pk.Input("dem")   # today
opt_c = gz.mask.ApplyMask(mask=gz.qa.MaskClouds(qa_band="QA60", bits=[10]))(opt)
                                                    # (B_o, H, W)             today
# σ⁰_dB = 10 log₁₀ σ⁰, after a 7 × 7 Lee filter
sar_db = LinearToDB()(gz.restore.DespeckleLee(window=7)(sar))
                                         # (B_s, H', W')  today + proposed (I)
fused = AlignAndStack(target="optical")(opt_c, sar_db, dem)
                                         # → (B_o + B_s + 1, H, W)  proposed (I)
crops = gz.learn.ModelOp(model=classifier)(fused)   # → (C, H, W)             today

g = pk.Graph(inputs={"optical": opt, "sar": sar, "dem": dem},
             outputs={"crop_map": crops, "fused": fused})    # today
```

**Notes.** The `Graph` topology is carrier-agnostic and works today, and the running example is a smaller instance of the same idea. Choosing the target grid decides what information is resampled away, so it stays a domain decision in geotoolz. The original gallery said that `Graph.summary()` catches a missing input at construction. However, it does not exist, and `Graph` checks for missing inputs when it is called ([Report 2](02_pipekit_surface.md#r2-group-a)).

(r9-ml)=
## Machine learning

(r9-case-3)=
### Case 3 · Training and sliding-window inference (geocatalog → pipekit-train → geopatcher)

**TL;DR.** Train on random chips from a catalog, then run the model over a whole scene in overlapping patches, with one preprocessing object on both paths.

**Problem.** Training draws chips $\mat{X}_k \in \mathbb{R}^{B \times 256 \times 256}$ for $k = 1, \ldots, K$. Inference splits a scene into $N_\text{patch}$ patches at a stride of 224 (32 pixels of overlap), predicts each patch $p$, and stitches the predictions by overlap-add ([Report 7](07_geopatcher.md#r7-axes)).

```{code-block} python
:caption: Case 3. Training and inference both use shipped code; the chip iterator is user code.
:linenos:

# Shapes: B bands; chips 256 × 256; K training chips; N_patch inference patches
import pipekit as pk
import geotoolz as gz
import geopatcher as gp
from geopatcher import spatial
from geopatcher.integrations.pipekit import ApplyToChips, GridSampler, MergePatches
from pipekit_train import IterableDataset, TrainingLoop

preprocess = pk.Sequential([                         # shared by both paths
    gz.radiometry.ToFloat32(),                # (B, H_win, W_win) → (B, H_win, W_win)  today
    gz.radiometry.PercentileClip(lower=2.0, upper=98.0),        # today
])
augment = pk.Sequential([
    gz.augment.RandomFlip(seed=0),                   # today
    gz.augment.RandomRotate90(seed=0),               # today
])

# Training: (augment ∘ preprocess)(X_k), y_k for k = 1 … K
train = IterableDataset(
    source=lambda: ((augment(preprocess(x)), y) for x, y in chips(catalog)),
    content_hash=catalog_hash)                       # today; chips() is user code
model, _ = TrainingLoop(model_op=gz.learn.ModelOp(model=net), dataset=train,
                        max_steps=10_000, batch_size=32).run()  # today; args sketched

# Inference: ŷ_n = Σ_p w_n^(p) ŷ_n^(p) / Σ_p w_n^(p)
patcher = gp.SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(256, 256)),     # today
    sampler=spatial.sampler.RegularStride(step=(224, 224)),     # today
    window=spatial.window.Hann(),                               # today
    aggregation=spatial.aggregation.OverlapAdd(),               # today
)
infer = pk.Sequential([
    GridSampler(patcher=patcher),                    # field → N_patch patches  today
    ApplyToChips(operator=preprocess | model),   # (B, 256²) → (1, 256²)  today
    MergePatches(aggregation=spatial.aggregation.OverlapAdd(),
                 domain=field.domain),               # N_patch patches → (1, H, W)  today
])
prediction = infer(gp.RasterField(scene))           # today
```

**Library map.** geocatalog (chip sources), pipekit (composition), pipekit-train (the loop and the dataset), geopatcher with its `[pipekit]` extra (patching), geotoolz (operators and `ModelOp`) and georeader (the field). **Notes.** Using the same `preprocess` object on both paths is the guarantee against train/serve skew, and it holds. Two names moved since the original gallery:
- `pa.ModelOp` became `gz.learn.ModelOp`, since pipekit-array has no `ModelOp`.
- `Stitch` became `MergePatches`.

The original also had `catalog.iter_random_slices`, which does not exist. Random chips come from geopatcher's `spatial.sampler.Random` over a `RasterField`, or from your own iterator, as here.

(r9-case-10)=
### Case 10 · Active learning (geopatcher → pipekit → geotoolz)

**TL;DR.** Score every chip by the model's uncertainty and label the hundred most uncertain ones first.

**Problem.** For the patches $p = 1, \ldots, N_\text{patch}$, we compute a per-patch score $\bar{\sigma}^{(p)} \in \mathbb{R}_{\ge 0}$, which is the mean predictive standard deviation over the patch's pixels $\mathcal{N}^{(p)}$, and we select the 100 patches with the largest $\bar{\sigma}^{(p)}$.

```{code-block} python
:caption: Case 10. The scoring and the patcher ship; the selection loop is proposal H.
:linenos:

# Shapes: N_patch patches of 256 × 256; 100 selected
import heapq
import numpy as np
import pipekit as pk
import geopatcher as gp
from geopatcher import spatial

score = pk.Sequential([
    preprocess,                                      # today
    gz.learn.ModelOp(model=uq_model, method="predict_std"),  # → (1, 256, 256)  today
    # σ̄^(p) = (1 / |𝒩^(p)|) Σ_{n∈𝒩^(p)} σ_n^(p)
    pk.Lambda(lambda s: float(np.nanmean(s)), name="mean"),  # → scalar     today
])
patcher = gp.SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(256, 256)),
    sampler=spatial.sampler.RegularStride(step=(256, 256)),   # no overlap
    window=spatial.window.Boxcar(),
    aggregation=spatial.aggregation.ByIndex(),   # per patch, no stitch  today
)
# the 100 largest σ̄^(p): the patches to label next
top = heapq.nlargest(100, patcher.split(field),
                     key=lambda p: score(p.data))    # user; or proposed (H)
```

**Notes.** `ByIndex` is right because the scores stay per patch rather than being stitched back into a field. The original used `pa.MeanScalar(field="sigma")`. However, pipekit-array's `MeanScalar` reduces over array axes (`axis=`) and has no `field` argument, so on a plain array `pipekit_array.MeanScalar()` works too. Scoring is the expensive part, which makes `geopatcher.run.parallel_map` the place to parallelise it.

(r9-serving)=
## Serving

(r9-case-4)=
### Case 4 · A LitServe inference API (georeader → pipekit → geotoolz)

**TL;DR.** Serve the plume pipeline behind a GPU endpoint that reads the requested scene and returns the flux with the pipeline's hash.

**Problem.** A request carries a scene URL and a bounding box. We read that window, run the pipeline and return the result together with an identifier that pins exactly which pipeline produced it.

```{code-block} python
:caption: Case 4. The read, the pipeline and the hash all exist today.
:linenos:

# Shapes: one request → one window X: (B, H_win, W_win)
import litserve as ls
import tempfile
from georeader.rasterio_reader import RasterioReader
from georeader.read import read_from_bounds
from pipekit_experiment import LocalModelRegistry
from pipekit_experiment.adapters.hydra import HydraConfigLoader

class PlumeAPI(ls.LitAPI):                            # user
    def setup(self, device):
        self.pipe = HydraConfigLoader.from_yaml(open("plume_v3.yaml").read())  # today
        # hash = SHA-256(canonical config): stable across processes
        self.hash = LocalModelRegistry(tempfile.mkdtemp()).store(self.pipe)    # today

    def decode_request(self, req):
        return read_from_bounds(RasterioReader(req["url"]), tuple(req["bbox"]))
                                                     # → GeoTensor (B, H_win, W_win)  today

    def predict(self, scene):
        return self.pipe(scene)                      # → flux dict             today

    def encode_response(self, out):
        return {**out, "pipeline_hash": self.hash}   # user

ls.LitServer(PlumeAPI(), accelerator="gpu").run(port=8000)   # user
```

**Notes.** The original's `georeader.read_from_url` and `pk.config_hash` do not exist. Reading is `read_from_bounds` on a `RasterioReader` (which opens URLs through GDAL), and the hash is the registry's content hash. Storing the same pipeline twice gives the same 64-character SHA-256, and changing one parameter changes it (we checked this). Proposal B would expose the hash without the throwaway registry. Heavy state lives in `setup` and never in `predict`, which is operational discipline rather than framework.

(r9-case-5)=
### Case 5 · A multi-pipeline backend (FastAPI)

**TL;DR.** Serve several named pipelines, each self-describing and pinned by hash, and let a promotion swap which one a name points at.

**Problem.** A map from names to pipelines. Each endpoint reads, runs and writes a COG, and the API lists every pipeline's config and hash.

```{code-block} python
:caption: Case 5. Registry tags give named, promotable pipelines today.
:linenos:

# Shapes: one request → X: (B, H_win, W_win) → COG
from fastapi import FastAPI
from georeader.save import save_cog
from pipekit_experiment import LocalModelRegistry

app = FastAPI()                                      # user
reg = LocalModelRegistry("registry/")                # today
# name ↦ hash, e.g. "plume" ↦ "9246b8…"; promotion = re-pointing the tag
for name in ("plume", "ndvi_etl"):
    PIPES[name] = reg.load(reg.resolve_tag(name))    # today (flat pipelines)

@app.post("/process/{name}")
def process(name: str, req: Request):                # user
    out = PIPES[name](read_window(req))              # read_window: case 4  today
    save_cog(out["enh"], f"s3://outputs/{req.id}.tif")   # → COG            today
    return {"cog": f"s3://outputs/{req.id}.tif", "hash": reg.resolve_tag(name)}

@app.get("/pipelines/{name}")
def describe(name: str):
    return PIPES[name].get_config()                  # self-documenting      today
```

**Notes.** `ConfigMixin` makes `get_config()` work for every operator, so the API documents itself. The registry's tags (`tag`, `resolve_tag`) are the promotion mechanism: we point `plume` at a new hash, and every server that resolves the tag picks it up. There is one limit, which is that `registry.load` rebuilds flat pipelines only, so a nested pipeline is stored for its hash and loaded from its YAML (proposal A closes the gap).

(r9-case-6)=
### Case 6 · User-uploaded pipelines, sandboxed

**TL;DR.** Let users paste a pipeline and run it, but only with operators on an allowlist.

**Problem.** Untrusted text in and a pipeline out, with every class it names checked against an allowlist before anything is instantiated.

```{code-block} python
:caption: Case 6. The allowlist ships for flat JSON; the YAML form is proposal F.
:linenos:

import pipekit as pk
import geotoolz as gz

# 𝒜 = {operator classes users may instantiate}
for cls in (gz.indices.NDVI, gz.radiometry.PercentileClip):
    pk.register(cls)                                 # today
pk.allowed()                                         # → ('geotoolz…NDVI', …)  today

try:
    pipe = pk.loads_sandboxed(user_text)             # flat JSON           today
    # pipe = loads_yaml_sandboxed(user_yaml)         # nested YAML      proposed (F)
except PermissionError as err:                       # not in 𝒜            today
    show_error(err)                                  # user
```

**Notes.** The allowlist is the right shape, since `loads_sandboxed` refuses any class that is not registered. There are two gaps:
- **Flat only.** The JSON form is flat, so a user pipeline is one operator until F walks a nested YAML and checks each `_target_`.
- **No introspection.** The original's `pk.registry.public_operators(...)` and `UntrustedTargetError` do not exist. Instead, you register classes explicitly, which is arguably safer.

`forbid_in_yaml` operators (closures) can never be loaded this way, which is the point. Resource limits are still the deployment's job.

(r9-case-7)=
### Case 7 · A tile server (z/x/y → PNG)

**TL;DR.** Render map tiles of a derived layer on demand.

**Problem.** For the tile `(z, x, y)`, we read the matching 256 × 256 window in Web Mercator, run the layer's pipeline and return a PNG.

```{code-block} python
:caption: Case 7. Reading and rendering ship; PNG encoding is user code.
:linenos:

# Shapes: tile (3, 256, 256) uint8
import mercantile
from georeader.read import read_from_bounds

@app.get("/tiles/{layer}/{z}/{x}/{y}.png")           # user (FastAPI)
def tile(layer: str, z: int, x: int, y: int):
    bounds = mercantile.xy_bounds(x, y, z)           # user
    gt = read_from_bounds(reader, bounds, crs_bounds="EPSG:3857")  # (B, H_win, W_win) today
    rgb = LAYERS[layer](gt)               # e.g. TrueColor | StretchToUint8  today
    return png_response(rgb.values)                  # user
```

**Notes.** geotoolz's `viz.TrueColor` and `viz.StretchToUint8` make the layer pipeline, whereas the original's `to_png_bytes` and `read_tile` helpers are not stack API. For an indexed archive, `geocatalog` finds the files under the tile and `geocatalog.load.load_raster` mosaics them. A tile cache in front matters more than Python speed.

(r9-ops)=
## Reproducibility and operations

(r9-case-9)=
### Case 9 · A pinned, hashed regulatory artifact (pipekit-experiment)

**TL;DR.** Freeze exactly how a published number was produced, and be able to rerun it years later.

**Problem.** We record the pipeline, the hashes of the models it uses, the dependency lock and the input identifiers in one file that can be checked and replayed.

```{code-block} python
:caption: Case 9. The artifact ships today, in pipekit-experiment rather than pipekit core.
:linenos:

from pipekit_experiment import InferenceArtifact, LocalModelRegistry
from pipekit_experiment.adapters.hydra import HydraConfigLoader

# At publication
yaml_text = open("plume_v3.yaml").read()
art = InferenceArtifact(                             # today
    pipeline_yaml=yaml_text,
    pinned_model_hashes={"classifier": reg.store(model, weights=w)},  # today
    backend_info={"python": "3.12", "image": "sha256:…"},
    deps_lock=open("pixi.lock").read(),
    metadata={"scene": "EMIT_L1B_RAD_…", "scene_sha256": "ab12…", "doi": "10.xxxx/…"},
)
art.save("artifacts/plume_2024Q3.json")              # today

# Years later
art = InferenceArtifact.load("artifacts/plume_2024Q3.json")      # today
pipe = HydraConfigLoader.from_yaml(art.pipeline_yaml)            # today
models = art.reload_models(reg)                      # today; weights round-trip
```

**Notes.** The original gallery put this in pipekit core as `pk.repro`. It shipped in pipekit-experiment instead, along with `TrainingArtifact` for the training side. Bit-exact reruns still need a container image hash, which is why `backend_info` exists. The operators inside can come from any library, because each one records its class path.

(r9-case-12)=
### Case 12 · Orchestrator task units (Prefect, Dagster)

**TL;DR.** Hand each scene to an orchestrator task that retries safely and writes to a path that depends only on its inputs.

**Problem.** One task per slice. A retry must produce the same output path, so the key is a function of (pipeline hash, slice) and is never random.

```{code-block} python
:caption: Case 12. The slice, loader and write ship; the hash helper is proposal B.
:linenos:

from prefect import flow, task                       # user
from geocatalog import open_catalog
from geocatalog.load import load_raster
from georeader.save import save_cog

@task(retries=3, retry_delay_seconds=60)
def run_one(pipe_yaml: str, sl, catalog_uri: str) -> str:
    pipe = HydraConfigLoader.from_yaml(pipe_yaml)    # today
    cat = open_catalog(catalog_uri)                  # today
    # key = hash(pipeline) ‖ hash(slice): identical on every retry
    key = f"{pipeline_hash(pipe)[:12]}-{hash(sl):x}"  # proposed (B)
    out = pipe(load_raster(cat, sl))                 # → GeoTensor             today
    save_cog(out, f"s3://daily/{key}.tif")           # idempotent by key       today
    return key

@flow
def daily(date: str):
    cat = open_catalog("s3://…/catalog.parquet")
    for sl in cat.iter_slices(resolution=(60.0, 60.0)):   # GeoSlice/row  today
        run_one.submit(PIPE_YAML, sl, "s3://…/catalog.parquet")
```

**Notes.** `GeoSlice` is a frozen dataclass, so it pickles to a worker and hashes stably. The original used `GeoSlice.from_dict` and `iter_slices_for_date`, neither of which exists, so we filter the catalog with `query(catalog, slice_)` for a date window first. The orchestrator owns retries and fan-out, whereas the stack owns what each task computes.

(r9-case-13)=
### Case 13 · Pipeline diffing and A/B regression

**TL;DR.** Before promoting a new pipeline version, show what changed in its config and how much its outputs moved.

**Problem.** Two pipelines, $f$ (current) and $f'$ (candidate). We report the config difference and, for a sample of $N_\text{scene}$ scenes, the per-scene $\mathrm{RMSE}_m = \Vert f(\mat{X}_m) - f'(\mat{X}_m) \Vert_2 / \sqrt{HW}$ and its distribution.

```{code-block} python
:caption: Case 13. Configs and the comparison loop work today; a packaged harness is proposal G.
:linenos:

# Shapes: N_scene sample scenes; outputs (H, W)
import numpy as np
from pipekit_experiment.adapters.hydra import HydraConfigLoader as H

old = H.from_yaml(open("plume_v3.yaml").read())      # today
new = H.from_yaml(open("plume_v4.yaml").read())      # today
diff = config_diff(old.get_config(), new.get_config())   # proposed (G)

# RMSE_m = ‖f(X_m) − f′(X_m)‖₂ / √(HW),  m = 1 … N_scene
def enh(pipe, x):
    return pipe(x)["enh"].values                    # (H, W)               today
rmse = np.array([np.sqrt(np.nanmean((enh(old, x) - enh(new, x)) ** 2))
                 for x in sample_scenes])            # (N_scene,)           user
```

**Notes.** A mean RMSE hides a bimodal regression, so we look at the whole $\{\mathrm{RMSE}_m\}$ distribution or, better, at the change in the quantity that matters ($\Delta Q_m$ for the plume). For xarray outputs, `xrtoolz.metrics.RMSE(variable, dims)` computes it over named dimensions. pipekit-evaluate's `Benchmark` is the structured version of this question for estimators against references, and extending it to "pipeline A against pipeline B" is proposal G.

(r9-geoapps)=
## New cases from the geoapps design

The geoapps platform (validation, attribution, watchlists, research) runs on this stack, and its design plan adds six shapes that the original gallery did not have. They matter because they stress the stack where real platforms live: versioned reruns, human verdicts and models that are retrained from them. The app-side pieces (the database, the queue, the UI) belong to geoapps, so the cases below show only the stack's half.

(r9-case-14)=
### Case 14 · A validation queue with priorities (pipekit → geotoolz)

**TL;DR.** Rank new predicted plumes so that analysts validate the most urgent ones first, and nothing unvalidated goes downstream.

**Problem.** Each predicted detection $d = 1, \ldots, N_\text{det}$ has a flux estimate $\hat{Q}_d \in \mathbb{R}_{\ge 0}$, an age $a_d \in \mathbb{R}_{\ge 0}$ and a flag $\delta_d \in \{0, 1\}$ (would validating it trigger a notification?). The priority $\pi_d \in \mathbb{R}_{\ge 0}$ combines them,

$$
\pi_d = g(\hat{Q}_d) (1 + \delta_d) \phi(a_d) ,
$$ (r9-eq-priority)

with the age weight

$$
\phi(a) = \begin{cases} 1 & a \le a_\text{notify} \\ e^{-(a - a_\text{notify}) / a_\text{decay}} & a > a_\text{notify} \end{cases}
$$ (r9-eq-priority-decay)

where $g$ increases with flux, $a_\text{notify} \in \mathbb{R}_{>0}$ is the notification window and $a_\text{decay} \in \mathbb{R}_{>0}$ is the decay after it. The code below computes {eq}`r9-eq-priority` from the running example's evidence.

```{code-block} python
:caption: Case 14. The stack produces each detection's evidence; the queue is geoapps.
:linenos:

# Shapes: N_det detections; each with (H, W) enhancement and mask
evidence = plume(radiance=scene)                     # running example     today
q_hat = evidence["flux"]["emission_rate_kg_s"]       # Q̂_d (kg s⁻¹)        today
sigma = evidence["flux"]["emission_rate_uncertainty_kg_s"]   # σ_Q        today
# π_d = g(Q̂_d) (1 + δ_d) φ(a_d)
priority = g(q_hat) * (1 + would_notify) * phi(age)  # geoapps-core       user
```

**Notes.** The stack's job ends at the evidence: an enhancement map, a mask, $\hat{Q}_d$ and its uncertainty $\sigma_Q$, all tied to the pipeline hash that made them. The platform's rule that a human has the final say is enforced in geoapps' database, not in the stack. Each verdict becomes a label anchored to (scene, geometry), which case 18 consumes.

(r9-case-15)=
### Case 15 · Scheduled pipeline families: discovery, monitoring, reanalysis, backfill

**TL;DR.** Run the same steps on four schedules, with every output keyed so that reruns are no-ops and a crashed backfill resumes.

**Problem.** A pipeline $f$ with a version tuple $v$ (pipeline hash, model hashes, input product versions). Each step's output is keyed by $\mathrm{key} = (\text{step}, \text{input id}, v)$. The families differ only in when they run, what they cover and where they write ([](#tbl-r9-families)):

:::{table} The four pipeline families. They share steps and keys, and differ in trigger, scope and stream.
:label: tbl-r9-families

| Family | Asks | Trigger | Writes to |
|---|---|---|---|
| Discovery | what is new anywhere? | a catalog poll | `nrt` |
| Monitoring | what happened at the sites we watch, including when nothing did? | each overpass of a known site | `nrt`, including non-detections |
| Reanalysis | what does the whole record say under one frozen $v$? | manual | `reanalysis-<id>` |
| Backfill | fill the gaps, with today's production $v$ | manual | `nrt`, gaps only |
:::

```{code-block} python
:caption: Case 15. Discovery and keyed steps on shipped code; the catalog driver is proposal C.
:linenos:

# Shapes: N_scene new scenes per poll
from geocatalog.sources import STACSource
from geocatalog.storage import CatalogBundle

# discovery: new items in an AOI and window, recorded with their query
bundle.ingest(STACSource("https://planetarycomputer…/api/stac/v1"),
              bounds=aoi, interval=window, collection="emit-l1b-rad")   # today
# key = (step, input id, v): rerun ⇒ no-op
v = (registry.store(plume_pipe), model_hash, product_version)   # today (registry)
for sl in bundle.catalog.iter_slices(resolution=(60.0, 60.0)):  # today
    key = (step, sl, v)                              # GeoSlice hashes stably  today
    if store.has(key): continue                      # geoapps job queue     user
    store.put(key, plume_pipe(load_raster(bundle.catalog, sl)))   # today
```

**Notes.** The stack supplies three things the families need:
- **Discovery with provenance.** `CatalogBundle.ingest` records the query that found each item, so a reanalysis can replay discovery ([Report 6](06_geocatalog.md#r6-beyond)).
- **Stable keys.** `GeoSlice` hashes, and the registry hashes pipelines and models.
- **Resumable patch runs.** `geopatcher.observe.PatchJournal` resumes a killed run inside one large scene.

Monitoring's non-detections ("the site was observable and nothing was seen") are what make a persistence estimate possible, and without them it is just a detection count. Reanalysis never overwrites, since it writes its own stream under one frozen $v$.

(r9-case-16)=
### Case 16 · What-if reruns in a sandbox

**TL;DR.** Rerun a published detection with one assumption changed (such as the wind product), and compare the two side by side without touching production.

**Problem.** We swap $U_\text{eff}$ (ERA5) for $U'_\text{eff}$ (GEOS-FP) with the mask held fixed. Since $Q = U_\text{eff} \mathrm{IME} / \ell$ ([plumax · 1](../geomodels/plumax/01_emission_rate.md#er1-ime)), the flux scales exactly:

$$
\frac{Q'}{Q} = \frac{U'_\text{eff}}{U_\text{eff}} .
$$ (r9-eq-whatif)

```{code-block} python
:caption: Case 16. The rerun and the exact check, from a real run.
:linenos:

# Shapes: (H, W) enhancement and mask, reused
# Q′/Q = U′_eff / U_eff, with 𝒫 fixed
q = {u: gz.plume.IMEEstimate(wind_speed=u)(mass_gt, mask_gt)["emission_rate_kg_s"]
     for u in (3.0, 4.5)}                            # today
q[4.5] / q[3.0]                    # → 1.5000000000000002   (= 4.5 / 3.0)
```

**Notes.** That ratio is an exact test oracle (gate J4 in the geoapps plan), and it holds on the shipped `IMEEstimate`. The stack makes what-ifs cheap because a pipeline is data, so we change one constructor argument, rehash, and the result lands under a new version tuple in a `sandbox/<user>/<name>` stream. The attribution can still flip under the new wind *direction*, which is the interesting part for analysts.

(r9-case-17)=
### Case 17 · Catalog explorer: preview, then ingest

**TL;DR.** Let an analyst browse a public catalog, preview a few scenes over their area, and ingest the collection only if it is worth it.

**Problem.** For an AOI and a time window, we list the matching items, load a cheap preview of one, and either stage the collection or discard it.

```{code-block} python
:caption: Case 17. Discovery, preview and staging all ship in geocatalog.
:linenos:

from geocatalog import GeoSlice, query
from geocatalog.load import load_raster
from geocatalog.sources import STACSource
from geocatalog.staging import stage

src = STACSource("https://earth-search.aws.element84.com/v1")   # today
rows = list(src.query(bounds=aoi, interval=window,
                      collection="sentinel-2-l2a", limit=20))   # 20 rows  today
# preview: one scene at 100 m instead of 10 m (100× fewer pixels)
preview = load_raster(catalog_of(rows), GeoSlice(bounds=aoi, interval=window,
                      resolution=(100.0, 100.0), crs="EPSG:32630"))   # today
if analyst_says_yes(preview):                        # geoapps UI           user
    stage(catalog_of(rows), dest="staging/", parallel=8)   # download + cache   today
```

**Notes.** "ETLs are user-callable" is the geoapps rule that drives this case. The stack's part is a cheap, lazy look before an expensive ingest, since the source returns `SourceRow`s without opening files and a coarse `GeoSlice` previews them. `catalog_of` stands for building an in-memory catalog from the rows (`geocatalog.sources.source_row_to_gdf_row`).

(r9-case-18)=
### Case 18 · The reinforcement loop: verdicts become labels, labels retrain the model

**TL;DR.** Every analyst verdict and every dismissed alert becomes a training label, a frozen label set retrains the detector, and a promotion re-points the production tag.

**Problem.** The label sets $\mathcal{D}_1 \subset \mathcal{D}_2 \subset \ldots$ grow as verdicts arrive. We retrain on one frozen set (such as $\mathcal{D}_{12}$, the `v12` of the code), compare the candidate with the production model on held-out scenes, and promote it only if it is better.

```{code-block} python
:caption: Case 18. Training, the registry and tags ship today; the label store is geoapps.
:linenos:

from pipekit_train import IterableDataset, TrainingLoop
from pipekit_experiment import LocalModelRegistry, TrainingArtifact

labels = frozen_label_set("v12")                     # 𝒟₁₂ from geoapps   user
ds = IterableDataset(source=lambda: chip_label_pairs(labels),
                     content_hash=labels.hash)       # today
candidate, _ = TrainingLoop(model_op=detector, dataset=ds).run()  # args sketched
ref = reg.store(candidate, weights=candidate.serialize_weights())   # today
if better(candidate, reg.load(reg.resolve_tag("plume-detector"))):  # user
    reg.tag(ref, "plume-detector", force=True)       # promotion             today
```

**Notes.** This is the loop that makes geoapps a model-reinforcement platform rather than a dashboard. The stack's contribution is content-addressing at every step: the label set's hash goes into the dataset, the dataset's hash and the model's hash go into a `TrainingArtifact`, and a tag names the production model. So a disputed detection can be traced to the exact labels its model was trained on. `serialize_weights` is `pipekit_jax.JaxModelOp`'s method, and other model wrappers bring their own.

(r9-case-19)=
### Case 19 · Event kinds as pipelines

**TL;DR.** Support plumes, oil spills, floods and cyclones with one platform, where each kind is a pipeline and a few rules and never a branch in the core.

**Problem.** A kind is a contract made of 1) the operators that turn a scene into detections (retrieve → detect → quantify), 2) the marks a detection carries, 3) a link rule that chains detections into events, and 4) the entities it can be attributed to.

```{code-block} text
:caption: The event-kind contract, with the stack's part in the first box.

  EventKind  (one per phenomenon; a plugin)
    ├── operators: retrieve → detect → quantify   ← a pipekit Graph
    ├── marks:     what a detection measures         (flux ± σ, area, peak, …)
    ├── link rule: detections → events            (IoU carry-over, a tracker)
    └── entities:  what it can be attributed to
            │
  shared core  (kind-agnostic): scene · detection · event · coverage · label · run
```

:::{table} The operator chain per kind, from shipped operators where they exist.
:label: tbl-r9-kinds

| Kind | Retrieve | Detect | Quantify |
|---|---|---|---|
| Methane plume | `matched_filter.MatchedFilter` / `ColumnEnhancement` | `plume.PlumeMask` | `plume.IMEEstimate`, `plume.CrossSectionalFlux` |
| Oil spill (SAR) | `restore.DespeckleLee` → dB (proposal I) | threshold + `segment` | area from the mask |
| Flood | `indices.NDWI` (optical) or dB backscatter | threshold + `segment` | flooded area, depth from a DEM |
| Heavy rain, cyclone | an xarray product through `xrtoolz.geo` | an `xrtoolz.geo` extremes operator | accumulation, track |
:::

**Notes.** Each kind's operator chain ([](#tbl-r9-kinds)) is a pipekit `Graph` like the running example, so the platform stores, hashes, reruns and compares every kind the same way. For methane, the retrieve and quantify steps are the ones that the [GeoModels boundaries](../geomodels/package_boundaries.md#pb-moves) move into radtrax and plumax. The link rule's IoU carry-over matches the stack's `geocatalog.matchup.IouAtLeast`, which pairs detections across overpasses. Rain and cyclone products are science-side xarray work rather than new app geometry, which is why xrtoolz carries them.

(r9-case-20)=
### Case 20 · Reanalysis evaluation against in-situ (geocatalog → xrtoolz)

**TL;DR.** Score a gridded reanalysis against station observations, pairing each observation with the nearest grid time.

**Problem.** We have observations $y_j \in \mathbb{R}$ at stations $j = 1, \ldots, N_\text{obs}$, with locations $\vect{s}_j \in \mathbb{R}^2$ and times $t_j$, and a gridded product $x(\vect{s}, t) \in \mathbb{R}$. We pair each $y_j$ with $x(\vect{s}_j, t)$ at the nearest time within 3 hours and compute $\mathrm{RMSE} = \sqrt{\frac{1}{N_\text{obs}} \sum_j (x(\vect{s}_j, t_j) - y_j)^2}$.

```{code-block} python
:caption: Case 20. Matchup, climatology and scoring all ship.
:linenos:

# Shapes: N_obs matched pairs; grid (time, lat, lon)
import pandas as pd
import pipekit as pk
from geocatalog.matchup import Intersects, NearestInTime, matchup
from xrtoolz import geo, metrics

# pairs = {(j, t) : s_j ∈ cell(t), |t − t_j| ≤ 3 h}
pairs = list(matchup(stations, reanalysis, spatial=Intersects(),
                     temporal=NearestInTime(dt=pd.Timedelta("3h"))))   # N_obs rows  today
anom = pk.Sequential([
    geo.ValidateCoords(),                            # (time, lat, lon)     today
    # x̃ = x − x̄_doy
    geo.RemoveClimatology(clim),                     # (time, lat, lon)     today
])
# RMSE = √((1/N_obs) Σ_j (x(s_j, t_j) − y_j)²)
rmse = metrics.RMSE(variable="t2m", dims="obs")     # today
score = rmse(paired_dataset(pairs, anom))            # user: assemble the pairs
```

**Notes.** This is the xarray side that the original gallery lacked, and it uses the same geocatalog and pipekit as the methane cases, with xrtoolz in place of geotoolz. `paired_dataset` stands for assembling the matched pairs into one `xr.Dataset` with an `obs` dimension, which is user code today. The original's hypothetical xrtoolz modules map onto real ones:
- `detrend` → `xrtoolz.geo`;
- `calc` → `xrgrad`;
- `transforms` → `xrtoolz.transforms`;
- `ice` → `xrtoolz.ice`, which is empty.

(r9-matrix)=
## Which libraries each case needs

:::{table} Cases against libraries. ✓ is a direct import; (✓) a transitive or optional use; P a dependency only through a proposal. The sister-packages column names which pipekit packages a case uses.
:label: tbl-r9-matrix

| Case | pipekit | Sister packages | geocatalog | geopatcher | geotoolz | xrtoolz | georeader |
|---|:-:|---|:-:|:-:|:-:|:-:|:-:|
| 1 Notebook | ✓ |  |  |  | ✓ |  | (✓) |
| 2 ETL | ✓ | experiment | ✓ |  | ✓ |  | (✓) |
| 3 Train + infer | ✓ | train | ✓ | ✓ | ✓ |  | ✓ |
| 4 LitServe | ✓ | experiment |  |  | ✓ |  | ✓ |
| 5 FastAPI | ✓ | experiment |  |  | ✓ |  | ✓ |
| 6 Sandboxed | ✓ | experiment (P) |  |  | ✓ |  |  |
| 7 Tile server | ✓ |  | (✓) |  | ✓ |  | ✓ |
| 8 QC | ✓ | array (P) |  |  | ✓ |  |  |
| 9 Artifact | (✓) | experiment |  |  | (✓) |  |  |
| 10 Active learning | ✓ | (array) |  | ✓ | ✓ |  | (✓) |
| 11 Fusion Graph | ✓ |  |  |  | ✓ |  |  |
| 12 Orchestrator | ✓ | experiment | ✓ |  | ✓ |  | ✓ |
| 13 A/B | ✓ | experiment, evaluate (P) |  |  | ✓ | (✓) |  |
| 14 Validation queue | ✓ | experiment |  |  | ✓ |  |  |
| 15 Pipeline families | ✓ | experiment | ✓ | (✓) | ✓ |  | (✓) |
| 16 What-if | ✓ | experiment |  |  | ✓ |  |  |
| 17 Explorer |  |  | ✓ |  |  |  | (✓) |
| 18 Reinforcement | ✓ | train, experiment, (evaluate) |  |  | ✓ |  |  |
| 19 Event kinds | ✓ |  | ✓ |  | ✓ | ✓ |  |
| 20 Reanalysis vs in-situ | ✓ |  | ✓ |  |  | ✓ |  |
:::

What the matrix ([](#tbl-r9-matrix)) says:

1. **pipekit is in 19 of 20 cases.** The exception is the explorer, which is pure catalog work, and it still belongs in a library of its own.
2. **pipekit-experiment is the busiest sister package (10 cases),** not pipekit-array (one optional use and one proposal), whereas the original gallery expected the reverse. Deployment needs hashing, YAML, registries and artifacts more than it needs array abstractions.
3. **geocatalog is in 8 cases, geopatcher in 3.** Both are used standalone, next to the domain libraries.
4. **xrtoolz now has real cases (13, 19, 20),** and they use the same framework and infrastructure as the methane cases. Swapping geotoolz for xrtoolz leaves the rest of each case intact, which is the strongest evidence that the framework layer is carrier-neutral.
5. **The geoapps cases (14 to 19) lean on hashing and the registry.** Versioned reruns, promotions and traceable labels are where a platform meets the stack.

(r9-patterns)=
## Patterns the cases reinforce

### pipekit owns composition; its sister packages own deployment infrastructure

:::{table} Where each concern lives at the pinned revisions, against the original gallery's plan.
:label: tbl-r9-ownership

| Concern | The gallery planned | It lives in |
|---|---|---|
| Operators, composition, parallel maps, flat JSON, allowlist | pipekit | pipekit |
| YAML loading | pipekit (`loads_yaml`) | pipekit-experiment (`HydraConfigLoader`) |
| Content hashing | pipekit (`config_hash`) | pipekit-experiment (the registry), plus `pipekit.hashing` primitives |
| Reproducibility artifacts | pipekit (`repro`) | pipekit-experiment (`InferenceArtifact`, `TrainingArtifact`) |
| Diffing and A/B | pipekit (`diff`) | not built (proposal G, pipekit-evaluate) |
| Training | — | pipekit-train |
| Spatiotemporal index, `GeoSlice` | geocatalog | geocatalog |
| Patches, `Field` / `Domain` | geopatcher | geopatcher |
| `ModelOp` | pipekit-array | geotoolz (`learn`), xrtoolz (`inference`), pipekit-jax (`JaxModelOp`); one family in [Report 16 B](16_geo_data_stack.md#r16-ws-b) |
| GeoTensor I/O | georeader | georeader |
:::

Each concern still has one home ([](#tbl-r9-ownership)), so the original's central claim holds, and the methane physics has its own owners in the [GeoModels package boundaries](../geomodels/package_boundaries.md#pb-rules). The difference is that pipekit core stayed small (about 3,200 lines), with the deployment infrastructure one package over. This answers the original's second risk, which was that pipekit would accrete infrastructure code.

### Three roles for georeader

Cases reach georeader in three ways ([](#tbl-r9-georeader)).

:::{table} How the cases reach georeader.
:label: tbl-r9-georeader

| Role | Cases | Pattern |
|---|---|---|
| Transitive, through `geocatalog.load.load_raster` | 1, 2, 3, 10, 15, 17 | the user never imports georeader |
| Direct read, request-driven | 4, 5, 7 | `georeader.read.read_from_bounds` on a `RasterioReader` |
| Direct write | 5, 12 | `georeader.save.save_cog` |
:::

### Parallelism is chosen per deployment

:::{table} The parallel primitive each deployment uses.
:label: tbl-r9-parallel

| Deployment | Primitive | Cases |
|---|---|---|
| Catalog-driven ETL | `pipekit.ProcessMap` | 2 |
| Per-patch work inside a scene | `geopatcher.run.parallel_map` | 3, 10 |
| GPU inference | none; the model batches | 4 |
| Async I/O inside a server | `pipekit.AsyncMap.run` (not `__call__`) | 5, 7 |
| Across machines | the orchestrator | 12, 15 |
:::

There is no automatic parallelism anywhere, by design, so each deployment picks its own primitive ([](#tbl-r9-parallel)).

### YAML is the lingua franca, read-only for now

Cases 2, 4, 9, 12 and 13 load pipelines from YAML, and every library's operators are reconstructible from `_target_` paths because they all subclass `pipekit.Operator`. So a YAML can name `geotoolz.plume.IMEEstimate` and `xrtoolz.geo.RemoveClimatology` side by side. Today the YAML is written by hand or generated with hydra-zen, and turning a built nested pipeline back into YAML is proposal A.

(r9-jax)=
## Where JAX helps

pipekit-jax has shipped, but not as the "JAX-traceable pipelines" library that the original gallery imagined ([Report 3](03_sister_libraries.md#r3-other)). It wraps an Equinox model as an operator with byte-identical weight round-trips, and it adds diffrax forward models for pipekit-cycle. For the cases here ([](#tbl-r9-jax)):

:::{table} The JAX benefit per case.
:label: tbl-r9-jax

| Case | Benefit | Status |
|---|---|---|
| 3 Train + infer | the model jits; patches are batched by `geopatcher.run.batch_split` and `stack_patches`, then vmapped | today, outside the pipeline |
| 4 LitServe | a jitted model inside an eager pipeline | today, model only |
| 10 Active learning | vmapped uncertainty over stacked patches | today, model only |
| 11 Fusion | a learnable alignment | needs traceable pipelines; not planned |
| 18 Reinforcement | weight round-trips through the registry by hash | today (`JaxModelOp`) |
:::

The original gallery's verdict stands: JAX changes how fast the inference-heavy cases run, but not how they are written. The pipeline itself is still eager Python.

(r9-risks)=
## Risks

- **pipekit-array is behind.** Only its first phase shipped. However, the cases need it in one place only (case 8, via proposal D), which is far less than planned. Fallback: `pk.AssertCallable` for the checks until phase C lands.
- **Infrastructure code accretes.** It did, but in pipekit-experiment rather than pipekit core, which kept core small. Fallback: keep each concern in its own module (`adapters.hydra`, `registry`, `artifacts`), as today.
- **Glue reinvented in each consumer.** `CatalogPipeline`-style glue (cases 2, 10, 12, 15) has no home, so every consumer writes its own loop. Recommended: proposal C, `geocatalog.integrations.pipekit`, mirroring geopatcher's integration module. Alternative: a small `pipekit-catalog` package.
- **YAML that only loads one way.** A team that builds pipelines in Python cannot save them as YAML, which breaks cases 2, 5, 9 and 12 for nested pipelines. Fallback: keep the YAML as the source of truth until proposal A lands.
- **Private API across packages.** geotoolz and geopatcher's pipekit bridge import `pipekit._base.operator` helpers, and making them public is [Report 16 §A](16_geo_data_stack.md#r16-ws-a).

(r9-open-questions)=
## Open questions

1. **Does deployment infrastructure stay in pipekit-experiment?** Hashing, YAML and artifacts are the busiest parts of the stack in these cases. Promoting a small `pipeline_hash` (proposal B) into pipekit core would make it reachable without the experiment package.
2. **Is the geoapps job queue a stack concern?** Case 15's keyed, resumable steps look general (any batch job wants them). Today they are geoapps' Postgres queue, so a small stack-level key-value "done" store might belong in pipekit-experiment.
3. **Should event kinds be expressible as configs?** If a kind's operator chain is a YAML `Graph`, then the platform can add kinds without code. That depends on proposal A and on a YAML form of `Graph`, which does not exist.
4. **How are human labels versioned?** Case 18 assumes frozen label sets with a content hash, which pipekit-train's datasets accept. So who owns the label store's hashing: geoapps or pipekit-experiment?

(r9-gates)=
## Testing gates for the proposals

Each proposal merges only when its check passes ([](#tbl-r9-gates)).

:::{table} One gate per proposal, each with a pass condition.
:label: tbl-r9-gates

| Gate | Proposal | Passes when |
|---|---|---|
| A1 YAML round-trip | A | `from_yaml(to_yaml(p))(x) == p(x)` for the running example's `Sequential`, and the two configs are equal |
| B1 Stable hash | B | `pipeline_hash(p) == registry.store(p)` for every case's pipeline, and it differs after any one parameter changes |
| C1 Catalog driver | C | a 100-slice catalog gives the same outputs at `n_workers=1` and `8`, and a rerun writes nothing new |
| D1 Array checks | D | each assert agrees with the NumPy reduction on NumPy and JAX inputs |
| E1 Geo checks | E | `AssertCRSEquals` and `AssertResolutionWithin` pass on matching and raise `QCError` on mismatched GeoTensors |
| F1 Sandboxed YAML | F | a nested YAML naming one unregistered `_target_` raises before any operator is built |
| G1 A/B harness | G | identical pipelines report an empty diff and zero RMSE on every scene |
| H1 Active sampler | H | its top-k equals `heapq.nlargest` on the same scores |
| J4 Wind swap (existing) | — | `Q′/Q = U′/U` to 1e-12 with the mask fixed; it passes today ({eq}`r9-eq-whatif`) |
:::

(r9-references)=
## References

### Sources

- The original use-case gallery, [`usecases.md`](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/plans/geotoolz/examples/usecases.md).
- The geoapps design plan and repository ([jejjohnson/geoapps](https://github.com/jejjohnson/geoapps)), for cases 14 to 19.
- The pinned code in `references/`, and the runs described in each caption.
