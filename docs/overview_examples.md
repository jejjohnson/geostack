---
title: Overview examples
---

Code that goes with the [GeoStack overview](overview.md). These are design
sketches of the intended APIs: several of the packages they use (for example
`statecatalog`, `pipekit-cycle`'s `DACycle` and `pipekit-train`) are planned
or still moving, so not every snippet runs against the libraries pinned in
`references/` today. Check the [capability index](api/capabilities.md) for
what exists now.

(ex-methane)=
## Methane attribution, end to end

The pipeline behind the [methane walkthrough](#fig-methane-walkthrough):
catalog the scenes, retrieve, match with in-situ, gap-fill, assimilate, index,
evaluate and register.

```python
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

# ───── Step 1: Catalog the scenes ─────
emit_cat = gc.open_catalog("s3://imeo/emit/2024/*.parquet")
in_situ_cat = gc.open_catalog("s3://imeo/tccon/2024/*.parquet")

# ───── Step 2 + 3: Per-scene retrieval pipeline ─────
retrieval = pk.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.radiometry.PercentileClip(p_min=2, p_max=98),
    gz.matched_filter.CH4MatchedFilter(target_spectrum=ch4_template),
])

# Or: a trained emulator instead of classical matched filter
registry = px.S3ModelRegistry(bucket="imeo-models")
neural_retrieval = registry.load("ch4_retrieval_v3")
retrieval_ml = pk.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.ModelOp(neural_retrieval),
])

# ───── Step 4: Gap-filling with UQ ─────
from pyrox_gp.adapters.pipekit import KrigingOperator

gap_fill = KrigingOperator(
    kernel="matern52",
    length_scale_km=50.0,
    return_uncertainty=True,
)

# ───── Step 5: DA cycle ─────
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

# ───── Step 6: State catalog output ─────
state_cat = sc.DuckDBStateCatalog.open("s3://imeo/methane_da_v3/states.parquet")
writer = sc.integrations.cycle.CycleStateWriter(
    state_catalog=state_cat,
    model_config_hash=pk.config_hash(da),
)

# Full forecasting pipeline
forecast_pipeline = pk.Sequential([da, writer])

# Run it
final_state, history = forecast_pipeline(initial_state, initial_carry_state)

# ───── Step 7: Multi-track evaluation ─────
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

# ───── Step 8: Register everything ─────
registry.attach_evaluation(model_hash=pk.config_hash(da), report=report)
```

## Tier examples

(ex-gaussx)=
### gaussX kernel matrix

```python
# Example: gaussX kernel matrix
from gaussx.kernels import Matern52
from gaussx.linalg import cholesky_solve

k = Matern52(length_scale=10.0, variance=1.0)
K = k.gram(x)                          # ←  Gram matrix as lineax operator
y_pred_mean = K @ cholesky_solve(K + sigma2 * I, y_train)
```

(ex-filterx)=
### filterX adapter as a DA analysis step

```python
# Example: filterX adapter as a DA analysis step
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
upstream. In geostack, use that operator; define a new one only when the
[capability index](api/capabilities.md) has nothing that fits, and add it to the
library that owns the concept.

```python
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

That’s it. `ConfigMixin` (inherited from Operator) auto-derives `get_config()` from the dataclass-like fields. The operator is pickleable, serialisable to YAML, registrable in the sandboxed loader, composable via `|`.

(ex-compose)=
### Compose pipelines

```python
# Linear: Sequential
preprocess = pk.Sequential([
    gz.radiometry.ToFloat32(),
    gz.radiometry.PercentileClip(p_min=2, p_max=98),
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
])

# Pipe syntax (equivalent)
preprocess = (
    gz.radiometry.ToFloat32()
    | gz.radiometry.PercentileClip(p_min=2, p_max=98)
    | gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11])
)

# DAG: Graph
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

(ex-catalogs)=
### Catalogs

```python
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
### Patcher (geopatcher)

```python
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

```python
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
### Swap classical forward model for a neural emulator

The key win of the architecture. Same DA cycle, different forward model. **This pattern is Steps 3→4 of the six-step cycle (§3.5)**: train an emulator from Step 1 simulations, then run Step 2’s inference loop with the emulator swapped in. Step 2 stays alive as the oracle that validated the emulator.

```python
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

# Later: use it in DA cycle
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

(ex-evaluation)=
### Evaluation pipeline

```python
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

```python
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
### Reproducibility artifact (regulatory)

```python
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
