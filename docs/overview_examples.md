---
title: Overview examples
subtitle: Code sketches for the patterns in the GeoStack overview
short_title: Examples
description: Code that goes with the GeoStack overview, from a full methane attribution pipeline to operators, catalogs, patchers, DA cycles, evaluation and reproducibility artifacts.
numbering:
  heading_2: true
  heading_3: true
  code: true
---

:::{warning} Design sketches
These snippets show the intended APIs. Several packages they use
(`statecatalog`, `pipekit-cycle`'s `DACycle`, `pipekit-train`) are planned or
still moving, so not every snippet runs against the libraries pinned in
`references/` today. Check the [capability index](api/capabilities.md) for
what exists now.
:::

Each section links back to the part of the [GeoStack overview](overview.md)
it illustrates.

(ex-methane)=
## Methane attribution, end to end

The pipeline behind {numref}`fig-methane-walkthrough`: catalog the scenes,
retrieve, gap-fill, assimilate, index, evaluate and register. Every stage is
an operator, so the whole thing composes, serialises and reruns.

```{code-block} python
:caption: Imports shared by every stage.
:linenos:

import pipekit as pk
import pipekit_cycle as pc
import pipekit_train as pt
import pipekit_evaluate as pe
import pipekit_experiment as px

import geocatalog as gc
import geopatcher as gp
import statecatalog as sc

import geotoolz as gz
from filterx.adapters.pipekit import EnKFAnalysis
from plumax.adapters.pipekit import PlumeForward, ColumnObs
```

::::{tab-set}

:::{tab-item} 1 · Catalog
```{code-block} python
:caption: Open the satellite and in-situ catalogs.
:linenos:

emit_cat = gc.open_catalog("s3://imeo/emit/2024/*.parquet")
in_situ_cat = gc.open_catalog("s3://imeo/tccon/2024/*.parquet")
```
:::

:::{tab-item} 2 · Retrieve
Classical retrieval and its learned replacement share the same slot in the
pipeline.

```{code-block} python
:caption: Classical retrieval: a matched filter.
:linenos:

retrieval = pk.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.radiometry.PercentileClip(p_min=2, p_max=98),
    gz.matched_filter.CH4MatchedFilter(target_spectrum=ch4_template),
])
```

```{code-block} python
:caption: Learned retrieval: a registered emulator.
:linenos:

registry = px.S3ModelRegistry(bucket="imeo-models")
neural_retrieval = registry.load("ch4_retrieval_v3")
retrieval_ml = pk.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.ModelOp(neural_retrieval),
])
```
:::

:::{tab-item} 3 · Gap-fill
```{code-block} python
:caption: Kriging with uncertainty, from pyrox-gp.
:linenos:

from pyrox_gp.adapters.pipekit import KrigingOperator

gap_fill = KrigingOperator(
    kernel="matern52",
    length_scale_km=50.0,
    return_uncertainty=True,
)
```
:::

:::{tab-item} 4 · Assimilate
```{code-block} python
:caption: An hourly DA cycle over the plume model.
:linenos:

da = pc.DACycle(
    forward_model=PlumeForward(species="CH4", dt=3600.0),
    obs_op=ColumnObs(instrument="TROPOMI"),
    analysis_step=EnKFAnalysis(inflation=1.05, localization="gaspari-cohn"),
    obs_source=pk.Sequential([
        gc.CatalogTimeQuery(window="hourly"),
        retrieval,
    ]),
    n_steps=24,
)
```
:::

:::{tab-item} 5 · Index
```{code-block} python
:caption: Write every analysed state into the state catalog.
:linenos:

state_cat = sc.DuckDBStateCatalog.open("s3://imeo/methane_da_v3/states.parquet")
writer = sc.integrations.cycle.CycleStateWriter(
    state_catalog=state_cat,
    model_config_hash=pk.config_hash(da),
)

# Full forecasting pipeline
forecast_pipeline = pk.Sequential([da, writer])

# Run it
final_state, history = forecast_pipeline(initial_state, initial_carry_state)
```
:::

:::{tab-item} 6 · Evaluate
```{code-block} python
:caption: Evaluate against CAMS, aircraft and TCCON at once.
:linenos:

evaluation = pe.Pipeline([
    pe.ByLeadTime(
        pe.ByRegion(
            regions={"permian": permian, "bakken": bakken, "global": None},
            metric=pe.metrics.pointwise.RMSE(),
        ),
    ),
    pe.metrics.probabilistic.CRPS(ensemble_dim="member"),
    pe.metrics.physical.MassClosure(tolerance=0.01),
    pe.ByEvent(
        detector=pe.events.MethanePlumeDetector(threshold_kg_h=100),
        metric=pe.metrics.detection.CSI(),
    ),
])

multitrack = pe.MultiTrackEvaluation(
    evaluation_pipeline=evaluation,
    references={
        "cams":     gc.open_catalog("s3://cams/methane_reanalysis/"),
        "aircraft": gc.open_catalog("s3://methaneair/2024/"),
        "tccon":    in_situ_cat,
    },
)

report = multitrack.run(predictions=history)
```
:::

:::{tab-item} 7 · Register
```{code-block} python
:caption: Attach the evaluation to the model that produced it.
:linenos:

registry.attach_evaluation(model_hash=pk.config_hash(da), report=report)
```
:::
::::

## Tier examples

(ex-gaussx)=
### gaussX kernel matrix

Tier 1 numerics ([briefs](#sec-tier-briefs)): kernels and solvers are
lineax operators, so the Gram matrix never has to be materialised.

```{code-block} python
:caption: A Matérn-5/2 GP posterior mean with gaussX.
:linenos:

from gaussx.kernels import Matern52
from gaussx.linalg import cholesky_solve

k = Matern52(length_scale=10.0, variance=1.0)
K = k.gram(x)                          # ←  Gram matrix as lineax operator
y_pred_mean = K @ cholesky_solve(K + sigma2 * I, y_train)
```

(ex-filterx)=
### filterX adapter as a DA analysis step

Tier 2 algorithms plug in through adapters. The analysis step satisfies
pipekit-cycle's `AnalysisStep` Protocol structurally: no base class needed.

```{code-block} python
:caption: An EnKF analysis step from filterX.
:linenos:

from filterx.adapters.pipekit import EnKFAnalysis

analysis = EnKFAnalysis(
    inflation=1.05,
    localization="gaspari-cohn",
    localization_radius_km=500.0,
)
# Satisfies pipekit_cycle.AnalysisStep Protocol via structural typing
```

## Key code patterns

(ex-operator)=
### Define an operator

A simplified sketch of how `geotoolz.radiometry.PercentileClip` is written
upstream.

:::{important} Reuse first
Use the upstream operator. Define a new one only when the
[capability index](api/capabilities.md) has nothing that fits, and add it to
the library that owns the concept.
:::

```{code-block} python
:caption: An operator is a class with fields and an _apply method.
:linenos:

from pipekit import Operator
from typing import ClassVar

class PercentileClip(Operator):
    """Clip a GeoTensor to [p_min, p_max] percentiles per band."""

    forbid_in_yaml: ClassVar[bool] = False
    p_min: float = 2.0
    p_max: float = 98.0
    axis: tuple = (-2, -1)

    def _apply(self, gt):
        import numpy as np
        lo = np.quantile(gt.data, self.p_min / 100, axis=self.axis, keepdims=True)
        hi = np.quantile(gt.data, self.p_max / 100, axis=self.axis, keepdims=True)
        return gt.with_data(np.clip(gt.data, lo, hi))
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

# Linear: Sequential
preprocess = pk.Sequential([
    gz.radiometry.ToFloat32(),
    gz.radiometry.PercentileClip(p_min=2, p_max=98),
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
])
```
:::

:::{tab-item} Pipe syntax
```{code-block} python
:caption: The same pipeline with the | operator.
:linenos:

preprocess = (
    gz.radiometry.ToFloat32()
    | gz.radiometry.PercentileClip(p_min=2, p_max=98)
    | gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11])
)
```
:::

:::{tab-item} Graph
```{code-block} python
:caption: A DAG fusing optical, SAR and elevation inputs.
:linenos:

opt = pk.Input("optical")
sar = pk.Input("sar")
dem = pk.Input("dem")

opt_clean = gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11])(opt)
sar_db    = gz.sar.LinearToDB()(gz.sar.LeeSpeckle(window=7)(sar))
fused     = gz.fusion.AlignAndStack(target_grid="optical")(opt_clean, sar_db, dem)
classify  = gz.ModelOp(crop_classifier)(fused)

g = pk.Graph(
    inputs={"optical": opt, "sar": sar, "dem": dem},
    outputs={"crop_map": classify, "fused_stack": fused},
)

result = g(optical=s2_gt, sar=s1_gt, dem=dem_gt)
# result["crop_map"], result["fused_stack"]
```
:::
::::

(ex-catalogs)=
### Catalogs

Observation catalogs index L0–L2 ([lifecycle](#sec-lifecycle)); matchup is
where the two streams converge ([two streams](#sec-two-streams)).

```{code-block} python
:caption: Build, open, query and match a geocatalog.
:linenos:

# Build a catalog from a directory of GeoTIFFs
import geocatalog as gc

cat = gc.build_raster_catalog(
    "s3://imeo/sentinel2/2024/*.tif",
    backend="duckdb",
    out_uri="s3://imeo/sentinel2_index_2024.parquet",
)

# Or load an existing one
cat = gc.open_catalog("s3://imeo/sentinel2_index_2024.parquet")

# Iterate slices for a query
for slice in cat.iter_slices(bbox=aoi, time=(t0, t1)):
    gt = gc.load_raster(slice)
    result = preprocess(gt)
    # ...

# Matchup with in-situ
in_situ_cat = gc.open_catalog("s3://imeo/tccon_index.parquet")

matched = gc.queries.matchup(
    primary=in_situ_cat,             # the sparse / accurate stream
    secondary=cat,                    # the dense satellite stream
    time_tolerance=pd.Timedelta(hours=3),
    space_tolerance_km=25.0,
)
for pair in matched:
    obs = gc.load_in_situ(pair.primary_slice)
    sat = gc.load_raster(pair.secondary_slice)
    # ... use as (label, feature) pairs
```

(ex-patcher)=
### Patcher

```{code-block} python
:caption: Four-axis patching around a model, for any carrier.
:linenos:

import geopatcher as gp
from geopatcher.integrations.pipekit import GridSampler, ApplyToChips, Stitch

# Four-axis patcher: geometry × sampler × window × aggregation
patcher = gp.SpatialPatcher(
    geometry=gp.SpatialRectangular((256, 256)),
    sampler=gp.SpatialRegularStride((224, 224)),
    window=gp.SpatialHann(),
    aggregation=gp.SpatialOverlapAdd(),
)

# Composed into a Sequential
inference = pk.Sequential([
    GridSampler(patcher),
    ApplyToChips(pk.Sequential([
        preprocess,
        gz.ModelOp(model, batch_size=8),
    ])),
    Stitch(gp.SpatialOverlapAdd()),
])

# Field adapter for whatever carrier you have
field = gp.RasterField(scene_reader)        # for GeoTensor
field = gp.XarrayField(scene_ds)             # for xarray Dataset
field = gp.XvecField(stations_ds)            # for point cubes
field = gp.GeoPandasField(zones_gdf)         # for vector

result = inference(field)
```

(ex-da-cycle)=
### Data assimilation cycle

Every ingredient satisfies a pipekit-cycle Protocol, so any forward model,
observation operator or analysis step that fits the Protocol can be swapped
in.

```{code-block} python
:caption: An ensemble DA cycle with explicit state carry.
:linenos:

import pipekit_cycle as pc
from filterx.adapters.pipekit import EnKFAnalysis
from plumax.adapters.pipekit import PlumeForward, ColumnObs

# All ingredients are Protocol-satisfying
forward = PlumeForward(species="CH4", dt=3600.0)         # ForwardModel
obs_op  = ColumnObs(instrument="TROPOMI")                # ObservationOperator
analysis = EnKFAnalysis(inflation=1.05)                   # AnalysisStep

# Compose into a DA cycle
da = pc.EnsembleDACycle(
    forward_model=forward,
    obs_op=obs_op,
    analysis_step=analysis,
    obs_source=obs_sequential,                            # pulls obs each step
    n_steps=24,
    n_members=40,
    save_history=True,
)

# State carry is explicit
initial_carry = pc.DAState(
    background_state=initial_state,
    ensemble_members=initial_ensemble,
    t=t0,
    cycle_count=0,
)

final_state, history = da(initial_state, initial_carry)
```

(ex-emulator)=
### Swap in a neural emulator

The key win of the architecture: the same DA cycle with a different forward
model. This is steps 3→4 of the [six-step cycle](#sec-six-steps): train an
emulator on step-1 simulations, then run step 2's inference loop with the
emulator swapped in. Step 2 stays alive as the oracle that validated it.

::::{tab-set}

:::{tab-item} Train the emulator
```{code-block} python
:caption: Train an emulator offline on simulations.
:linenos:

# Train an emulator offline
emulator_loop = pt.TrainingLoop(
    model_op=gz.ModelOp(ChemistryEmulatorNet()),
    dataset=pt.SimulationDataset(
        forward_model=PlumeForward(species="CH4", dt=3600.0),
        prior=AtmosphericPrior(distribution="climatology"),
        n_samples=10_000,
    ),
    loss=pt.MSE(),
    optimizer_config={"name": "adam", "lr": 1e-3},
    n_epochs=100,
    backend="lightning",
    callbacks=[pt.LogToExperiment(tracker=mlflow_tracker)],
)
emulator_op = emulator_loop.run()
registry.store(emulator_op, name="methane_emulator_v3")
```
:::

:::{tab-item} Swap it in
```{code-block} python
:caption: The same DA cycle; only the forward model changes.
:linenos:
:emphasize-lines: 8
neural_forward = pc.NeuralForward(
    model_op=registry.load("methane_emulator_v3"),
    dt=3600.0,
)

# Same DA cycle, just swap the forward model
da_emulated = pc.EnsembleDACycle(
    forward_model=neural_forward,              # ← swapped
    obs_op=obs_op,                              # same
    analysis_step=analysis,                     # same
    obs_source=obs_sequential,                  # same
    n_steps=24,
    n_members=40,
)
# Target: ~100× faster forecasts (not yet benchmarked); same composability
```
:::
::::

(ex-evaluation)=
### Evaluation pipeline

```{code-block} python
:caption: Multi-track evaluation whose report is a content-addressed artifact.
:linenos:

import pipekit_evaluate as pe

evaluation = pe.Pipeline([
    # Field × Point-wise, decomposed by region and lead time
    pe.ByLeadTime(
        pe.ByRegion(
            regions={"permian": permian_mask, "global": None},
            metric=pe.metrics.pointwise.RMSE(),
        ),
    ),
    # Field × Spectral
    pe.metrics.spectral.PSDCompare(
        spatial_dims=("lat", "lon"),
        expected_slope=-3.0,
    ),
    # Field × Probabilistic
    pe.metrics.probabilistic.CRPS(ensemble_dim="member"),
    # Event × Detection
    pe.ByEvent(
        detector=pe.events.MethanePlumeDetector(threshold_kg_h=100),
        metric=pe.metrics.detection.CSI(),
    ),
    # Budget × Physical-constraint
    pe.metrics.physical.MassClosure(tolerance=0.01),
])

# Multi-track: vs reanalysis, vs analysis, vs observations
multitrack = pe.MultiTrackEvaluation(
    evaluation_pipeline=evaluation,
    references={
        "cams": cams_cat,
        "tccon": tccon_cat,
        "aircraft": methaneair_cat,
    },
)
report = multitrack.run(predictions=forecast_history)

# Result is a content-addressed artifact
print(report.to_pandas())
report.save("s3://imeo/eval/methane_v3_2024Q3.report")

# Diff against the previous version
old = pe.EvaluationReport.load("s3://imeo/eval/methane_v2_2024Q3.report")
diff = report.diff(old)
```

(ex-benchmark)=
### Benchmark contract

```{code-block} python
:caption: A pre-registerable, content-addressed benchmark contract.
:linenos:

from pipekit_evaluate.benchmark import BenchmarkContract

contract = BenchmarkContract(
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
    splitter_config={
        "_target_": "pipekit_train.splitters.SpatioTemporalBlockSplit",
        "spatial_block_km": 200,
        "temporal_block_days": 60,
    },
    block_size_spatial_km=200,
    block_size_temporal_days=60,
    metric_configs=[...],
    baselines=["pipekit_evaluate.baselines.OptimalInterpolationBaseline"],
    max_test_evaluations=1,
)

# Content-addressed, pre-registerable
hash = contract.content_hash()
contract.to_yaml().save("s3://oceanbench/contracts/v1.0.0.yaml")
```

(ex-repro)=
### Reproducibility artifact

```{code-block} python
:caption: Freeze a pipeline, its inputs and its lockfile for a cold rerun.
:linenos:

import pipekit as pk

# Freeze a pipeline + its inputs + lockfile
artifact = pk.repro.freeze(
    pipeline=da_emulated,
    inputs={
        "scene_catalog_uri": "s3://imeo/emit/2024_Q3_frozen/",
        "scene_catalog_sha256": "ab12cd34...",
    },
    deps_lock="pixi.lock",
    metadata={"author": "ej", "doi": "10.xxxx/mars-2024Q3"},
)
artifact.save("/regulatory/mars_methane_2024Q3.gtar")

# Years later
art = pk.repro.load("/regulatory/mars_methane_2024Q3.gtar")
assert art.pipeline_hash == "expected-hash-from-paper"
result = art.rerun()                # cold-rerun, no cloud accounts needed
```
