---
title: Overview examples
subtitle: Code sketches for the patterns in the GeoStack overview
short_title: Examples
description: Code that goes with the GeoStack overview, from a full methane attribution pipeline to operators, catalogs, patchers, DA cycles, evaluation and reproducibility artifacts.
numbering:
  heading_1: true
  heading_2: true
  code: true
---

:::{warning} Design sketches
These snippets show the intended APIs. Every call into the stack carries a
status marker, so you can tell what runs today from what is planned:

- `today`: the name is in the [capability index](api/capabilities.md), at
  the versions it lists (pipekit 0.0.2, geotoolz 0.10.2 and siblings).
  `today*` means the name exists but the arguments are sketched.
- `proposed (X)`: not in the index yet. The letter is the workstream that
  adds it ([the decision](overview.md#sec-decision)).
- `unverified`: lives in a library outside the index (gaussx, filterax,
  plumax, pyrox), so its API was not checked here.
:::

Each section links back to the part of the [GeoStack overview](overview.md)
it illustrates. Every sketch follows the same conventions: a shapes header
names the dimensions, the equation sits in a comment above the line that
computes it, and each line ends with its input → output shapes.

(ex-methane)=
## Methane attribution, end to end

The pipeline behind {numref}`fig-methane-walkthrough`: catalog the scenes,
retrieve, gap-fill, assimilate, index, evaluate and register. Every stage is
an operator, so the whole thing composes, serialises and reruns. The
problem each stage solves, with its equations, is in the
[walkthrough](overview.md#sec-walkthrough).

```{code-block} python
:caption: Imports shared by every stage.
:linenos:

import pipekit as pk                                         # today
import pipekit_cycle as pc                                   # today
import pipekit_train as pt                                   # today
import pipekit_evaluate as pe                                # today
import pipekit_experiment as px                              # today

import geocatalog as gc  # today (ships as geotoolz-catalog)
import geopatcher as gp  # today (ships as geotoolz-patcher)
import geotoolz as gz                                        # today
import statecatalog as sc                                    # proposed (B)

from filterax.adapters.pipekit import EnKFAnalysis           # unverified
from plumax.adapters.pipekit import PlumeForward, ColumnObs  # unverified
```

::::{tab-set}

:::{tab-item} 1 · Catalog
```{code-block} python
:caption: Open the satellite and in-situ catalogs (stage 1).
:linenos:

# Shapes: one GeoSlice per scene or station record
emit_cat = gc.open_catalog("s3://imeo/emit.parquet")    # → GeoCatalog  today
tccon_cat = gc.open_catalog("s3://imeo/tccon.parquet")  # → GeoCatalog  today

# scenes = {s : footprint(s) ∩ aoi ≠ ∅, t₀ ≤ tₛ ≤ t₁}
scenes = gc.query(emit_cat, bbox=aoi, time=(t0, t1))  # → rows  today*
```
:::

:::{tab-item} 2 · Retrieve
Classical retrieval and its learned replacement share the same slot in the
pipeline. `MaskClouds` builds the mask and `ApplyMask` applies it, so the
retrieval is a small graph rather than a chain.

```{code-block} python
:caption: Classical retrieval: a cloud-masked matched filter (stage 1).
:linenos:

# Shapes: B bands, H × W pixels
scene = pk.Input("scene")  # GeoTensor (B, H, W)  today

# cᵢ = 1 if the QA bits flag pixel i as cloud
clouds = gz.MaskClouds(qa_band=-1, bits=[10, 11])(scene)  # → (H, W)  today
clear = gz.ApplyMask()(scene, clouds)  # (B, H, W) → (B, H, W)  today*

# αᵢ = (xᵢ − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t), the estimate   [today*]
alpha = gz.MatchedFilter(target=ch4_template)(clear)  # (B, H, W) → (H, W)

retrieval = pk.Graph(inputs={"scene": scene}, outputs={"alpha": alpha})  # today
```

```{code-block} python
:caption: Learned retrieval: a registered model in the same slot.
:linenos:

# Shapes: B bands, H × W pixels
registry = px.S3ModelRegistry(bucket="imeo-models")  # today*
net = registry.load("ch4_retrieval_v3")              # content-addressed  today

# α = g_φ(x), trained with the matched filter as its oracle
alpha_ml = gz.ModelOp(net)(clear)       # (B, H, W) → (H, W)  today
```
:::

:::{tab-item} 3 · Gap-fill
```{code-block} python
:caption: Kriging with uncertainty, from pyrox (stage 3).
:linenos:

# Shapes: M observations, N grid cells
from pyrox.adapters.pipekit import KrigingOperator  # unverified

# m = K_*f (K_ff + σ² I)⁻¹ y  [unverified]
# v = diag(K_** − K_*f (K_ff + σ² I)⁻¹ K_f*)
gap_fill = KrigingOperator(                         # (M,) → (N,), (N,)
    kernel="matern52",
    length_scale_km=50.0,
    return_uncertainty=True,
)
```
:::

:::{tab-item} 4 · Assimilate
The EnKF analysis needs an ensemble, so the cycle is an `EnsembleDACycle`
with 40 members. The observation operator matches the instrument the
observations come from: EMIT. `retrieval` maps one scene to one map, so the
cycle's observation source wraps it: each hour it queries the catalog for
that hour's scenes, loads them and retrieves each one.

```{code-block} python
:caption: An hourly ensemble DA cycle over the plume model (stage 4).
:linenos:

# Shapes: N state, M observations, N_e = 40 members, T = 24 steps
# yₜ = {α(s) : s an EMIT scene with t ≤ tₛ < t + 1 h}
retrieve_one = pk.Lambda(lambda gt: retrieval(scene=gt)["alpha"])  # today
hourly_obs = pk.Sequential([                      # today
    gc.CatalogTimeQuery(emit_cat, window="1h"),   # t → GeoSlices  proposed (B)
    pk.ThreadMap(gc.LoadRaster() | retrieve_one), # → S × (H, W)  today*
])

# xᵇₜ⁽ⁱ⁾ = 𝓜(xᵃₜ₋₁⁽ⁱ⁾)   [unverified]
forward = PlumeForward(species="CH4", dt=3600.0)  # (N_e, N) → (N_e, N)
# y_pred = H x for EMIT columns, as in obs_source   [unverified]
obs_op = ColumnObs(instrument="EMIT")             # (N_e, N) → (N_e, M)
# xᵃ⁽ⁱ⁾ = xᵇ⁽ⁱ⁾ + K (y + ε⁽ⁱ⁾ − H xᵇ⁽ⁱ⁾), ε⁽ⁱ⁾ ~ 𝒩(0, R)  [unverified]
analysis = EnKFAnalysis(                          # (N_e, N) → (N_e, N)
    inflation=1.05, localization="gaspari-cohn"
)

da = pc.EnsembleDACycle(  # T × (forecast, analysis)  today*
    forward_model=forward,
    obs_op=obs_op,
    analysis_step=analysis,
    obs_source=hourly_obs,                        # EMIT scenes, each hour
    n_steps=24,
    n_members=40,
)
da_hash = pk.config_hash(da)  # content hash of the config  proposed (A)
```
:::

:::{tab-item} 5 · Index
```{code-block} python
:caption: Write every analysed state into the state catalog (stage 5).
:linenos:

# Shapes: T steps, N_e members, N cells
state_cat = sc.DuckDBStateCatalog.open(               # proposed (B)
    "s3://imeo/methane_da_v3/states.parquet"
)
writer = sc.CycleStateWriter(                         # proposed (B)
    state_catalog=state_cat, model_config_hash=da_hash
)

forecast_pipeline = pk.Sequential([da, writer])       # today
final_state, history = forecast_pipeline(x0, carry0)  # history: (T, N_e, N)
```
:::

:::{tab-item} 6 · Evaluate
pipekit-evaluate already ships `Task`, `Estimator`, `Oracle`, `Scorer` and
`Benchmark`. The region, lead-time and multi-track helpers below are
proposed on top of them (an [open question](overview.md#sec-open-questions)).

```{code-block} python
:caption: Evaluate against CAMS, aircraft and TCCON at once (stage 5).
:linenos:

# Shapes: T lead times, N_e members, N cells, R regions
evaluation = pe.Pipeline([                   # proposed (F)
    pe.ByLeadTime(pe.ByRegion(               # (T, N) → (T, R)  proposed (F)
        regions={"permian": permian, "bakken": bakken, "global": None},
        metric=pe.metrics.RMSE(),
    )),
    # CRPS(F, y) = 𝔼|X − y| − ½ 𝔼|X − X′|   [proposed (F)]
    pe.metrics.CRPS(ensemble_dim="member"),  # (T, N_e, N) → (T, N)
    # |∑ᵢ xᵢ − ∑ᵢ rᵢ| / ∑ᵢ rᵢ ≤ 0.01, r the reference
    pe.metrics.MassClosure(tolerance=0.01),  # (T, N) → (T,)  proposed (F)
])

report = pe.MultiTrackEvaluation(            # proposed (F)
    evaluation_pipeline=evaluation,
    references={"cams": cams_cat, "aircraft": aircraft_cat, "tccon": tccon_cat},
).run(predictions=history)                   # → EvaluationReport
```
:::

:::{tab-item} 7 · Register
```{code-block} python
:caption: Attach the evaluation to the model that produced it (stage 5).
:linenos:

registry.attach_evaluation(model_hash=da_hash, report=report)  # proposed (E)
```
:::
::::

## Tier examples

(ex-gaussx)=
### gaussx kernel matrix

Tier 1 numerics ([briefs](overview.md#sec-tier-briefs)): kernels and solvers
are lineax operators, so the Gram matrix never has to be materialised.

```{code-block} python
:caption: A Matérn-5/2 GP posterior mean with gaussx.
:linenos:

# Shapes: M training points, d input dimensions
from gaussx.kernels import Matern52                 # unverified
from gaussx.linalg import cholesky_solve            # unverified

k = Matern52(length_scale=10.0, variance=1.0)
# K = [k(xᵢ, xⱼ)]ᵢⱼ
K = k.gram(x)  # (M, d) → (M, M) operator, never dense
# m = K (K + σ² I)⁻¹ y
mean = K @ cholesky_solve(K + sigma2 * I, y_train)  # (M,) → (M,)
```

(ex-filterax)=
### filterax adapter as a DA analysis step

Tier 2 algorithms plug in through adapters. The analysis step satisfies
pipekit-cycle's `AnalysisStep` protocol structurally: no base class needed.
pipekit-cycle already ships `EnsembleInflator` and `CovarianceLocalizer`, so
the adapter can take those objects instead of string options.

```{code-block} python
:caption: An EnKF analysis step from filterax.
:linenos:

# Shapes: N state, N_e members
from filterax.adapters.pipekit import EnKFAnalysis  # unverified

# Pᵇ ← ρ ∘ (λ Pᵇ), λ = 1.05, ρ a Gaspari–Cohn taper of 500 km
analysis = EnKFAnalysis(                            # (N_e, N) → (N_e, N)
    inflation=1.05,
    localization="gaspari-cohn",
    localization_radius_km=500.0,
)
assert isinstance(analysis, pc.AnalysisStep)        # the protocol: today
```

## Key code patterns

(ex-operator)=
### Define an operator

A simplified sketch of how `geotoolz.PercentileClip` is written upstream.

:::{important} Reuse first
Use the upstream operator. Define a new one only when the
[capability index](api/capabilities.md) has nothing that fits, and add it to
the library that owns the concept.
:::

```{code-block} python
:caption: An operator is a class with fields and an _apply method.
:linenos:

from typing import ClassVar

import numpy as np
from pipekit import Operator                           # today


class PercentileClip(Operator):
    """Clip a GeoTensor to [p_min, p_max] percentiles per band."""

    forbid_in_yaml: ClassVar[bool] = False
    p_min: float = 2.0
    p_max: float = 98.0
    axis: tuple = (-2, -1)

    def _apply(self, gt):
        # Shapes: B bands, H × W pixels
        # q_lo = Q_{p_min}(x_b), q_hi = Q_{p_max}(x_b) for each band b
        q = dict(axis=self.axis, keepdims=True)
        lo = np.quantile(gt.data, self.p_min / 100, **q)  # → (B, 1, 1)
        hi = np.quantile(gt.data, self.p_max / 100, **q)  # → (B, 1, 1)
        # x_clip = min(max(x, q_lo), q_hi)
        return gt.with_data(np.clip(gt.data, lo, hi))  # (B, H, W) → (B, H, W)
```

`ConfigMixin`, inherited from `Operator`, derives `get_config()` from the
fields. The operator is picklable, serialisable to YAML, registrable in the
sandboxed loader and composable with `|`.

(ex-compose)=
### Compose pipelines

::::{tab-set}

:::{tab-item} Sequential
```{code-block} python
:caption: A linear pipeline.
:linenos:

# Shapes: B bands, H × W pixels; every step maps (B, H, W) → (B, H, W)
preprocess = pk.Sequential([               # today
    gz.ToFloat32(),                        # today
    gz.PercentileClip(p_min=2, p_max=98),  # today
])
```
:::

:::{tab-item} Pipe syntax
```{code-block} python
:caption: The same pipeline with the | operator.
:linenos:

preprocess = gz.ToFloat32() | gz.PercentileClip(p_min=2, p_max=98)  # today
```
:::

:::{tab-item} Graph
```{code-block} python
:caption: A DAG fusing optical, SAR and elevation inputs.
:linenos:

# Shapes: B optical bands, P SAR polarisations, H × W optical grid
opt = pk.Input("optical")                               # (B, H, W)  today
sar = pk.Input("sar")                                   # (P, H′, W′)  today
dem = pk.Input("dem")                                   # (1, H″, W″)  today

# σ⁰_dB = 10 log₁₀ σ⁰, after speckle filtering   [proposed (C)]
despeckled = gz.sar.LeeSpeckle(window=7)(sar)           # same shape
sar_db = gz.sar.LinearToDB()(despeckled)  # same shape  proposed (C)
# regrid SAR and DEM onto the optical grid, then stack along bands
align = gz.fusion.AlignAndStack(target_grid="optical")  # proposed (C)
fused = align(opt, sar_db, dem)                         # → (B + P + 1, H, W)
crop_map = gz.ModelOp(crop_classifier)(fused)  # → (H, W)  today

g = pk.Graph(                                           # today
    inputs={"optical": opt, "sar": sar, "dem": dem},
    outputs={"crop_map": crop_map, "fused_stack": fused},
)
result = g(optical=s2_gt, sar=s1_gt, dem=dem_gt)        # dict of GeoTensors
```
:::
::::

(ex-catalogs)=
### Catalogs

Observation catalogs index L0–L2 ([lifecycle](overview.md#sec-lifecycle));
matchup is where the two streams converge
([two streams](overview.md#sec-two-streams)).

```{code-block} python
:caption: Build, open, query and match a geocatalog.
:linenos:

# Shapes: one GeoSlice per file; M matched pairs
cat = gc.build_raster_catalog(                                   # proposed (B)
    "s3://imeo/sentinel2/2024/*.tif",
    backend="duckdb",
    out_uri="s3://imeo/sentinel2_index_2024.parquet",
)
cat = gc.open_catalog("s3://imeo/sentinel2_index_2024.parquet")  # today

# rows with footprint ∩ aoi ≠ ∅ and t₀ ≤ t ≤ t₁
rows = gc.query(cat, bbox=aoi, time=(t0, t1))                    # today*

# Wⱼ = {k : |tⱼ − tₖ| ≤ 3 h, d(sⱼ, sₖ) ≤ 25 km}
# k⋆(j) = argmin over Wⱼ of (d / 25 km)² + (|Δt| / 3 h)²
matched = gc.queries.matchup(  # → M ≤ |tccon| pairs  proposed (B)
    primary=tccon_cat,  # the sparse, accurate stream
    secondary=cat,  # the dense satellite stream
    time_tolerance=pd.Timedelta(hours=3),
    space_tolerance_km=25.0,
    select="nearest",  # one k per j; ties: earlier time, then index
)
```

`gc.intersect` (today) already pairs overlapping rows across two catalogs;
matchup adds the time and distance tolerances on top of it.

(ex-patcher)=
### Patcher

```{code-block} python
:caption: Four-axis patching around a model, for any carrier.
:linenos:

# Shapes: C channels, H × W scene; 256² patches, stride 224
from geopatcher.integrations.pipekit import (     # proposed (C)
    ApplyToChips,
    GridSampler,
)

patcher = gp.SpatialPatcher(                       # today*
    geometry=gp.SpatialRectangular((256, 256)),
    sampler=gp.SpatialRegularStride((224, 224)),
    window=gp.SpatialHann(),
    aggregation=gp.SpatialOverlapAdd(),
)
field = gp.RasterField(scene_reader)               # GeoTensor field  today

# y(s) = ∑ₚ wₚ(s) f(xₚ)(s) / ∑ₚ wₚ(s), wₚ the Hann window of p
inference = pk.Sequential([                        # today
    GridSampler(patcher),  # (C, H, W) → P × (C, 256, 256)  proposed (C)
    ApplyToChips(preprocess | gz.ModelOp(model)),  # per patch  proposed (C)
    gz.Stitch(aggregation=gp.SpatialOverlapAdd()),  # → (K, H, W)  today*
])
result = inference(field)
```

The stitch step is geotoolz's existing `Stitch`, which already joins
georeferenced tiles; only the sampler and the per-patch wrapper are new.

(ex-da-cycle)=
### Data assimilation cycle

Every ingredient satisfies a pipekit-cycle protocol, so any forward model,
observation operator or analysis step that fits can be swapped in. This
standalone example assimilates TROPOMI, so its observation operator and its
observation source both name TROPOMI.

```{code-block} python
:caption: An ensemble DA cycle with explicit state carry.
:linenos:

# Shapes: N state, M observations, N_e = 40 members, T = 24 steps
forward = PlumeForward(species="CH4", dt=3600.0)  # ForwardModel  unverified
obs_op = ColumnObs(instrument="TROPOMI")  # ObservationOperator  unverified
analysis = EnKFAnalysis(inflation=1.05)           # AnalysisStep  unverified

da = pc.EnsembleDACycle(                          # today*
    forward_model=forward,
    obs_op=obs_op,
    analysis_step=analysis,
    obs_source=tropomi_source,                    # TROPOMI columns, each hour
    n_steps=24,
    n_members=40,
    save_history=True,
)

carry0 = pc.DAState(                              # today*
    background_state=x0,                          # (N,)
    ensemble_members=X0,                          # (N_e, N)
    t=t0,
    cycle_count=0,
)
final_state, history = da(x0, carry0)             # history: (T, N_e, N)
```

(ex-emulator)=
### Swap in a neural emulator

The key win of the architecture: the same DA cycle with a different forward
model. This is steps 3→4 of the
[six-step cycle](overview.md#sec-six-steps): train an emulator on step-1
simulations, then run step 2's inference loop with the emulator swapped in.
Step 2 stays alive as the oracle that validated it.

::::{tab-set}

:::{tab-item} Train the emulator
```{code-block} python
:caption: Train an emulator offline on simulations.
:linenos:

# Shapes: S = 10,000 simulations, N state
# φ⋆ = argmin_φ 𝔼_θ ‖f(θ) − f_φ(θ)‖², f_φ the emulator
loop = pt.TrainingLoop(                                      # today*
    model_op=gz.ModelOp(ChemistryEmulatorNet()),
    dataset=pt.SimulationDataset(  # S × (θ, f(θ))  today
        forward_model=PlumeForward(species="CH4", dt=3600.0),
        prior=AtmosphericPrior(distribution="climatology"),
        n_samples=10_000,
    ),
    loss=pt.MSE(),                                           # today
    optimizer_config={"name": "adam", "lr": 1e-3},
    n_epochs=100,
    backend="lightning",
    callbacks=[pt.LogToExperiment(tracker=mlflow_tracker)],  # today
)
emulator_op = loop.run()
registry.store(emulator_op, name="methane_emulator_v3")  # today
```
:::

:::{tab-item} Swap it in
```{code-block} python
:caption: The same DA cycle; only the forward model changes.
:linenos:
:emphasize-lines: 7

# xᵇₜ = f_φ⋆(xᵃₜ₋₁), the trained emulator
neural_forward = pc.NeuralForward(  # (N_e, N) → (N_e, N)  today
    model_op=registry.load("methane_emulator_v3"),
    dt=3600.0,
)
da_emulated = pc.EnsembleDACycle(
    forward_model=neural_forward,   # ← swapped
    obs_op=obs_op,                  # same
    analysis_step=analysis,         # same
    obs_source=tropomi_source,      # same
    n_steps=24,
    n_members=40,
)
# Target: ~100× faster forecasts. Not yet benchmarked.
```
:::
::::

(ex-evaluation)=
### Evaluation pipeline

What exists today is the benchmark cube: a `Task` (prior, simulator,
observation operator, data), an `Estimator` for each rung of the six-step
cycle, an `Oracle` (usually rung 2) and a `Scorer`. The multi-track helpers
sketched in the methane pipeline should be built on these.

```{code-block} python
:caption: Score an emulator-based estimator against the classical oracle.
:linenos:

# Shapes: θ ∈ ℝᵈ parameters, y ∈ ℝᴹ observations
task = pe.Task(                                          # today*
    prior=prior, simulator=forward, obs_op=obs_op, data=tccon_cat
)
oracle = pe.Oracle(estimator=classical_da)  # rung 2: ground truth  today*
candidate = pe.Estimator(fn=emulated_da)    # rung 4  today*

# CRPS(F, y) = 𝔼|X − y| − ½ 𝔼|X − X′|   [today*]
bench = pe.Benchmark(                                                          #
    task=task, oracle=oracle, estimators=[candidate], scorer=crps
)
report = bench.run()                        # → BenchmarkReport  today
```

(ex-benchmark)=
### Benchmark contract

```{code-block} python
:caption: A pre-registerable, content-addressed benchmark contract.
:linenos:

# Shapes: (T, lat, lon) in and out; 200 km × 60 day blocks for the splits
contract = pe.BenchmarkContract(  # proposed (F)
    name="oceanbench_ssh_2025",
    version="1.0.0",
    task="gap_filling",
    input_carrier_spec={"shape": ("time", "lat", "lon"), "dtype": "float32"},
    output_carrier_spec={"shape": ("time", "lat", "lon"), "dtype": "float32"},
    variant="deterministic",
    reference_catalogs={
        "reanalysis": "s3://glorys/2023/index.parquet",
        "analysis": "s3://duacs/2023/index.parquet",
        "observations": "s3://altimetry/2023/index.parquet",
    },
    train_catalog_uri="s3://oceanbench/train.parquet",
    val_catalog_uri="s3://oceanbench/val.parquet",
    test_catalog_uri="s3://oceanbench/test.parquet",
    splitter_config={             # the one place block sizes live
        "_target_": "pipekit_train.splitters.SpatioTemporalBlockSplit",
        "spatial_block_km": 200,
        "temporal_block_days": 60,
    },
    metric_configs=[...],
    baselines=["pipekit_evaluate.baselines.OptimalInterpolationBaseline"],
    max_test_evaluations=1,
)
digest = contract.content_hash()  # pre-registerable
contract.to_yaml().save("s3://oceanbench/contracts/v1.0.0.yaml")
```

The earlier draft also set `block_size_spatial_km` and
`block_size_temporal_days` beside `splitter_config`. They now live only in
the splitter config, so the two cannot disagree.

(ex-repro)=
### Reproducibility artifact

pipekit-experiment already ships `InferenceArtifact`, a reproducibility
snapshot for an inference pipeline. The freeze-and-rerun flow below is
proposed as an extension of it, not as a second mechanism.

```{code-block} python
:caption: Freeze a pipeline, its inputs and its lockfile for a cold rerun.
:linenos:

artifact = px.InferenceArtifact(                       # today*
    pipeline=da_emulated,
    inputs={
        "scene_catalog_uri": "s3://imeo/emit/2024_Q3_frozen/",
        "scene_catalog_sha256": "ab12cd34...",
    },
    deps_lock="pixi.lock",
    metadata={"author": "ej", "doi": "10.xxxx/mars-2024Q3"},
)
artifact.save("/regulatory/mars_methane_2024Q3.gtar")  # proposed (E)

# Years later: a cold rerun, no cloud accounts needed   [proposed (E)]
art = px.InferenceArtifact.load("/regulatory/mars_methane_2024Q3.gtar")  #
assert art.pipeline_hash == "expected-hash-from-paper"
result = art.rerun()                                   # proposed (E)
```
