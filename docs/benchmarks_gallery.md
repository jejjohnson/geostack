---
title: Benchmarks gallery
subtitle: Eighteen draft benchmark contracts in five domains, checked against the stack as it is
short_title: Benchmarks gallery
description: Worked examples for Report 15. Each entry is a draft benchmark contract (task, variant, splits, references by track, baselines, scorers and known failure modes) with a content hash, and every stack piece it cites is checked against the pinned code. The splits follow Report 15's corrected rule, and the stack names follow Reports 11 to 15.
keywords:
  - benchmark
  - contract
  - ocean
  - land
  - atmosphere
  - remote sensing
  - methane
numbering:
  heading_1: true
  heading_2: true
  table: true
  figure: true
---

+++ {"part": "abstract"}

[Report 15](reports/15_benchmarking.md) says what makes a benchmark a
benchmark: a frozen contract with a hash. This gallery writes eighteen of
them, across the ocean, the land, the atmosphere, remote sensing and one
mathematical-model chain, so the framework can be judged on concrete cases.
Each entry is a draft contract, specific enough to pre-register after a
domain expert's review. Three things set this version apart from its source
draft. **The splits follow Report 15's rule**: the held-out set reproduces
the gap the product will face in use, so most random space-time blocks
become withheld platforms or causal splits, and in eleven entries a "split"
the source names (season, climate phase, altitude, time of day) is a
breakdown of the score instead. **The stack names are
the real ones**: each entry lists what imports from the pinned code today
(61 distinct pieces across the gallery), what a report proposes or the
overview defers (18), and what nobody has proposed yet (6). **Every claim about the gallery is
computed**: the notebook `notebooks/reports/benchmarks_gallery.py` holds
the contracts as data, hashes them, checks every stack name and asserts
every count on this page that describes the contracts.

+++

:::{note} About this page
:class: dropdown

| | |
|---|---|
| **Status** | Draft contracts, not pre-registered. Originally the companion gallery to the Report 15 scoping report; the domain content is kept, the splits and the stack mapping are rewritten |
| **Reading time** | about 35 minutes; each entry stands alone |
| **Audience** | anyone designing a benchmark for one of these variables, or checking whether the stack can run one |
| **Companions** | [Report 15](reports/15_benchmarking.md) (the contract, the splits, the baselines); [Report 14](reports/14_pipekit_evaluate.md) (scorers, wrappers, events); [Report 13](reports/13_state_catalog.md) (analyses and reanalyses as catalog rows); [Report 11](reports/11_pipekit_train.md) (training datasets) |
| **Code checked against** | pipekit 0.0.2 at `ef3c17f`, xrtoolz 0.0.5 at `308c7f2`, geotoolz 0.10.2 with geocatalog 0.5.1 and geopatcher at `088475e`. Read on 13 October 2026 |
| **Checks** | `notebooks/reports/benchmarks_gallery.py` imports every name an entry marks as today, checks that every reserved name raises, checks that every proposed name with a target module does not exist there yet, checks that every declared split has the piece that makes it, hashes each contract, asserts the counts on this page that describe the contracts, and draws [](#fig-gallery-readiness) |
| **Reference datasets** | named as the source names them. Resolutions and band counts are kept only where they are well established; a pre-registered contract replaces each name with a catalog URI and its hash |
:::

:::{important} What changed from the source draft
**Counts.** The source says six domains and, in different places, 22 and 23
benchmarks. It lists eighteen entries in five domains, one of them (5.1) a
chain of five stages; this page counts eighteen entries, and writes 5.1 as a
cascade of six contracts ([](#gal-5-1)). **Tasks.** Each entry names its task as a level
transition ([Report 15](reports/15_benchmarking.md#r15-framework)). Four
remote-sensing entries are not transitions up the ladder: two map L1 to L1
between instruments, and two simulate L1 data, from an atmospheric state
(4.3) or from a high-resolution truth through the instrument (4.4). So a
contract's task has to name its carriers, not only its levels. **Splits.** Nine source entries prescribe
`SpatioTemporalBlockSplit` and six more a temporal block. After Report 15's
correction, none needs random space-time blocks: mapping withholds a platform, forecasting is causal, and
transfer withholds a named region. **Stratifications.** In eleven entries the source lists season, ENSO
or NAO phase, altitude, elevation or latitude band, ocean basin, surface type,
spectral band, sensor combination or time of day as splits; they are decompositions of the score ([Report 14](reports/14_pipekit_evaluate.md#r14-ws-c)'s
wrappers) and move to their own row. **Names.** `statecatalog` is
geocatalog's state profile (Report 13), `EventCatalog` is an event profile
(Report 14), `MultiTrackEvaluation` is the runner with several references
(Report 15), and `MultiStageEvaluation` becomes a cascade contract
([](#gal-5-1)). Each entry has a **Differs** box where its own content changed.
:::

(gal-reading)=
## How to read an entry

Each entry is one contract. Its rows are the fields of Report 15's proposed
`BenchmarkContract` ([workstream A](reports/15_benchmarking.md#r15-ws-a)),
plus the two the gallery adds to make a draft reviewable.

:::{table} The rows of an entry and the contract fields they fill.
:label: tbl-gal-rows

| Row | Contract field | What it holds |
|---|---|---|
| Task | `task` | the level transition (L0 unstructured observations to L4 reanalysis or forecast), with a name |
| Variant | `variant` | deterministic, probabilistic or both; a probabilistic contract refuses point entries |
| Gap the split reproduces | `splits.gap` | the situation in use the held-out set imitates ([Report 15, stage 2](reports/15_benchmarking.md#r15-stage-2)) |
| Splits | `splits` | withhold by a column (platform, station, float, scene, sensor, source class, campaign), causal, or withhold a region |
| Scores stratified by | `scorers` wrappers | groups the score is broken down by; not splits |
| References, by track | `references` | one product per track; a missing track is reported unavailable |
| Baselines | `baselines` | the shared implementations every entry is compared with |
| Scorers | `scorers` | grouped by the lens of [Report 14](reports/14_pipekit_evaluate.md#r14-framework) |
| Known failure modes | (gallery) | what models typically get wrong; what the scorers must be able to see |
| Contract hash | `content_hash()` | the first 12 hex digits of the SHA-256 of the entry's key-sorted JSON, from the notebook |
:::

The **Stack** line under each table sorts the pieces the entry needs by
status: **today** (imports from the module named, at the pinned revision),
**name reserved** (imports, but raises `NotImplementedError` until xrtoolz's
V5), **proposed** (by the report and workstream named), **not proposed yet**
(the gallery's own findings, collected in [](#gal-needs)), **external** (a tool
outside the stack) and **not pinned** (a stack library that is not in
`references/`, so not checked). Every entry also needs Report 15's
`BenchmarkContract` (A1) and the runner's several references (D2); they are
left off the per-entry lines.


(gal-ocean)=
## Ocean

The most mature domain for machine-learning benchmarks: a community benchmark (OceanBench), strong reference products (GLORYS, DUACS, OSTIA, ISAS, CCI), and known gaps in the subsurface, on the shelves and in biogeochemistry. Each entry says which gap it addresses.


(gal-1-1)=
### 1.1 · Sea surface height

SSH sets the surface geostrophic flow, and mesoscale eddies carry most of the ocean's variability. Wide-swath altimetry (SWOT) changes the sampling that decades of optimal-interpolation maps were built on. OceanBench's first edition [@johnson2023oceanbench] poses this task on real data, and [Report 15](reports/15_benchmarking.md) runs its synthetic twin.

:::{table} Contract 1.1: Sea surface height.
:label: tbl-gal-1-1

| Term | Value |
|---|---|
| Task | L1 → L2, gap filling |
| Variant | both |
| Gap the split reproduces | mapping where one altimetry mission flew, from the others |
| Splits | withhold by `mission`: one nadir mission; causal: held-out year |
| Scores stratified by | region (western boundary currents, coastal) |
| References, by track | reanalysis: GLORYS12; analysis: DUACS L4; observations: withheld along-track mission; SWOT where available |
| Baselines | OI (L, τ fitted to the training years); persistence (forecast variant); training-period climatology |
| Scorers | RMSE of SSH and of geostrophic velocity; PSD score, resolved scale; geostrophic balance error; eddy detection (POD, FAR); CRPS, spread-skill (ensemble variant) |
| Known failure modes | double penalty rewards smoothed maps; spectra too steep at small scales; western boundary currents under-resolved; coastal cells worst; SWOT cross-track artefacts learned as signal |
| Contract hash | `867c6d7f7bc2…` |
:::

**Stack.** **Today:** `RMSE` (xrtoolz.metrics), `PSDScore` (xrtoolz.metrics), `resolved_scale` (xrtoolz.metrics), `GeostrophicBalanceError` (xrtoolz.metrics), `GeostrophicVelocities` (xrtoolz.ocn), `CalculateSSHAlongtrack` (xrtoolz.ocn), `along_track_psd_score` (xrtoolz.metrics), `OkuboWeiss` (xrtoolz.ocn), `CRPS` (xrtoolz.metrics), `SpreadSkillRatio` (xrtoolz.metrics). **Name reserved, raises:** `ProbabilityOfDetection`, `FalseAlarmRatio`. **Proposed:** `optimal_interpolation` (Report 15 C1), `persistence` (Report 15 C2), `ClimatologyForecast` (Report 15 C3), `split_by` (Report 15 B2), `split_causal` (Report 15 B2), `build_state_catalog` (Report 13 A2). **External:** py-eddy-tracker (eddy tracking).

:::{warning} Differs from the source
The source prescribes 200 km × 60 day blocks and, in the same entry, "split by mission, not pooled". [Report 15](reports/15_benchmarking.md#r15-stage-2) shows the mission split is the one that reproduces a mapping product's error (2.70 cm, against 3.83 cm with the blocks), so it is the contract's split. The source's OI baseline uses τ = 15 days while its splits assume 30; the contract fits both scales to the training years instead.
:::

(gal-1-2)=
### 1.2 · Sea surface temperature

SST is the most observed ocean variable, used from weather forecasting to ENSO monitoring. Clouds and the diurnal cycle are the hard parts, and every L4 product is used downstream, so a biased gap-filler spreads.

:::{table} Contract 1.2: Sea surface temperature.
:label: tbl-gal-1-2

| Term | Value |
|---|---|
| Task | L1 → L2, gap filling |
| Variant | deterministic |
| Gap the split reproduces | filling cloud gaps: real cloud masks from other days laid over clear scenes |
| Splits | mask transplant: cloud masks of other days; withhold by `platform`: withheld drifters; causal: held-out year |
| Scores stratified by | time of day; front regions |
| References, by track | analysis: OSTIA, ESA CCI SST, MUR, OISST; observations: drifters, near-surface Argo, ship lines |
| Baselines | OI (L, τ fitted to the training years); training-period climatology; persistence with climatology fill |
| Scorers | RMSE, bias; PSD score; SSIM, FSS; marine heatwave detection; gradient magnitude near fronts |
| Known failure modes | clear-sky bias in training; diurnal aliasing of polar orbiters; fronts smoothed; tidal coastal variability missed |
| Contract hash | `2c6b783b5d20…` |
:::

**Stack.** **Today:** `RMSE` (xrtoolz.metrics), `Bias` (xrtoolz.metrics), `PSDScore` (xrtoolz.metrics), `SSIM` (xrtoolz.metrics), `GradientDifference` (xrtoolz.metrics), `SimulatedClouds` (geotoolz.augment). **Name reserved, raises:** `ProbabilityOfDetection`, `DurationError`. **Proposed:** `optimal_interpolation` (Report 15 C1), `ClimatologyForecast` (Report 15 C3), `persistence` (Report 15 C2), `FSS` (Report 14 B1), `MarineHeatwaveDetector` (Report 14 E1), `split_by` (Report 15 B2), `split_causal` (Report 15 B2), `ProfileSlice` (overview, deferred).

:::{warning} Differs from the source
The source splits by 100 km × 30 day blocks. An SST gap-filler is used to fill cloud gaps, so the held-out set is clear-sky pixels hidden under real cloud masks from other days (geotoolz's `SimulatedClouds` draws synthetic ones today), plus withheld drifters. Balancing the time of day is a stratification of the score, not a split.
:::

(gal-1-3)=
### 1.3 · Sea surface salinity

Salinity drives the haline circulation and marks river plumes, ice melt and rain. Satellite salinity has a low signal-to-noise ratio: calibration errors can match the natural variability, so uncertainty matters more than for SST.

:::{table} Contract 1.3: Sea surface salinity.
:label: tbl-gal-1-3

| Term | Value |
|---|---|
| Task | L1 → L2, gap filling |
| Variant | probabilistic |
| Gap the split reproduces | mapping where a held-out Argo float measured; a river system for transfer |
| Splits | withhold by `float_id`: withheld floats; withhold a region: one major river system |
| Scores stratified by | region (river plumes, polar) |
| References, by track | analysis: CCI SSS, CMEMS L4 SSS, ISAS; observations: near-surface Argo salinity, thermosalinograph lines |
| Baselines | World Ocean Atlas climatology; OI (L, τ fitted to the training years); persistence, weekly |
| Scorers | RMSE, bias; CRPS; river-plume detection; T–S consistency; regional means against ISAS |
| Known failure modes | satellite retrieval errors spatially structured; near-surface Argo gradient unresolved; river-plume edges sub-daily; ice contamination at high latitude |
| Contract hash | `0e47b9ba77fd…` |
:::

**Stack.** **Today:** `RMSE` (xrtoolz.metrics), `Bias` (xrtoolz.metrics), `CRPS` (xrtoolz.metrics), `density_from_ts` (xrtoolz.ocn), `EvaluateByRegion` (xrtoolz.metrics). **Proposed:** `ClimatologyForecast` (Report 15 C3), `optimal_interpolation` (Report 15 C1), `persistence` (Report 15 C2), `ThresholdEventDetector` (Report 14 E1), `split_by` (Report 15 B2), `split_regions` (Report 15 B2), `ProfileSlice` (overview, deferred). **Not pinned:** pyrox-gp (Gaussian-process gap filling).

:::{warning} Differs from the source
The 300 km × 90 day blocks become withheld floats, which is the gap a salinity map fills. Leaving out a river system is kept, as the transfer test it is.
:::

(gal-1-4)=
### 1.4 · Ocean colour

Chlorophyll is the most used satellite biogeochemical variable and the base of the marine food web. The retrieval is dominated by atmospheric correction, and coastal waters with sediment and dissolved organic matter break the standard band-ratio algorithms.

:::{table} Contract 1.4: Ocean colour.
:label: tbl-gal-1-4

| Term | Value |
|---|---|
| Task | L1 → L2, gap filling |
| Variant | deterministic |
| Gap the split reproduces | a biogeochemical province not seen in training; in-situ by cruise |
| Splits | withhold a region: one Longhurst province; withhold by `cruise`: withheld cruises; causal: held-out years |
| Scores stratified by | water type (open ocean, coastal); season |
| References, by track | analysis: OC-CCI, GlobColour; observations: HPLC chlorophyll (SeaBASS, NOMAD); operational: NASA OC4, GIOP |
| Baselines | NASA OC4 band ratio; GIOP; monthly OC-CCI climatology |
| Scorers | RMSE of log10 chlorophyll, bias by water type; PSD score; bloom onset (POD, FAR with timing tolerance); distribution distance of log chlorophyll; front alignment with SST |
| Known failure modes | atmospheric correction in coastal waters; sun glint; cloud shadows; sediment-rich water overestimated; models learn the bias |
| Contract hash | `40840af4f114…` |
:::

**Stack.** **Today:** `BandRatio` (geotoolz.spectral), `RMSE` (xrtoolz.metrics), `Bias` (xrtoolz.metrics), `PSDScore` (xrtoolz.metrics), `Wasserstein1` (xrtoolz.metrics). **Name reserved, raises:** `ProbabilityOfDetection`, `FalseAlarmRatio`. **Proposed:** `ClimatologyForecast` (Report 15 C3), `EvaluateByGroup` (Report 14 C1), `split_regions` (Report 15 B2), `split_by` (Report 15 B2), `split_causal` (Report 15 B2), `ProfileSlice` (overview, deferred).

:::{warning} Differs from the source
The source's province split and its temporal block are kept, the second as a causal split. Withholding cruises is added here: in-situ HPLC samples come by cruise, and two samples from one cruise are not independent. The structural lens in the source (front alignment with SST) needs the SST product of entry 1.2 as a second input, so the contract names that product's hash.
:::

(gal-1-5)=
### 1.5 · Ocean biogeochemistry

BGC-Argo has only recently grown large enough for gridded products; three-dimensional biogeochemistry is the source's largest ocean gap. Sensor drift makes uncertainty part of the answer, not an extra.

:::{table} Contract 1.5: Ocean biogeochemistry.
:label: tbl-gal-1-5

| Term | Value |
|---|---|
| Task | L0 → L2, discretization and gap filling |
| Variant | probabilistic |
| Gap the split reproduces | a float's whole record, unseen in training |
| Splits | withhold by `float_id`: withheld floats; causal: held-out years |
| Scores stratified by | basin; depth |
| References, by track | climatology: WOA; analysis: CMEMS BGC reanalysis; observations: BGC-Argo floats, GLODAPv2 bottles |
| Baselines | WOA monthly climatology; OI (slow scales); ESPER |
| Scorers | RMSE per variable and depth; CRPS per depth, calibration on bottles; stoichiometric consistency; oxygen-minimum-zone and DCM detection |
| Known failure modes | sensor calibration drift sets the floor; near-surface optics, thermocline; very low oxygen overestimated; sparse floats south of 60° S |
| Contract hash | `8fb617842097…` |
:::

**Stack.** **Today:** `RMSE` (xrtoolz.metrics), `CRPS` (xrtoolz.metrics), `ReliabilityCurve` (xrtoolz.metrics), `DensityInversionFraction` (xrtoolz.metrics). **Proposed:** `ClimatologyForecast` (Report 15 C3), `optimal_interpolation` (Report 15 C1), `ThresholdEventDetector` (Report 14 E1), `split_by` (Report 15 B2), `split_causal` (Report 15 B2), `ProfileSlice` (overview, deferred). **External:** ESPER (empirical seawater property estimates). **Not pinned:** pyrox-gp (3-D Gaussian-process gap filling).

:::{warning} Differs from the source
Leave-one-float-out is kept: it is the source's headline split and the right one. The source's split by ocean basin becomes a stratification of the score.
:::


(gal-land)=
## Land

Land-surface variables have a long meteorological record (ERA5-Land, GHCN, IMERG), but machine-learning benchmarks for them lag weather benchmarks. The entries bring the same multi-track, multi-lens discipline.


(gal-2-1)=
### 2.1 · Near-surface temperature

Heatwaves kill, land surface temperature is the most used land remote-sensing variable, and 2 m temperature is the headline public forecast. The station network is uneven and skin temperature is not air temperature, which makes the references disagree with each other.

:::{table} Contract 2.1: Near-surface temperature.
:label: tbl-gal-2-1

| Term | Value |
|---|---|
| Task | L1 → L2 mapping, and L3 → L4 forecast |
| Variant | both |
| Gap the split reproduces | mapping at withheld stations; forecasting the future |
| Splits | withhold by `station`: withheld stations; causal: held-out years; withhold a region: one Köppen–Geiger zone |
| Scores stratified by | elevation band; climate zone |
| References, by track | reanalysis: ERA5-Land, MERRA-2; observations: GHCN-Daily stations; satellite: MODIS, VIIRS, SLSTR land surface temperature |
| Baselines | persistence (24 h, 168 h); training-period climatology; operational forecast where licensed |
| Scorers | RMSE, MAE, bias; CRPS, rank histogram; diurnal phase and amplitude; heatwave detection and duration; bias against elevation |
| Known failure modes | urban heat islands (absent from ERA5-Land too); lapse-rate bias in terrain; uneven station network; diurnal phase over deserts and snow; skin against 2 m temperature |
| Contract hash | `cbe972a8f817…` |
:::

**Stack.** **Today:** `RMSE` (xrtoolz.metrics), `MAE` (xrtoolz.metrics), `Bias` (xrtoolz.metrics), `CRPS` (xrtoolz.metrics), `RankHistogram` (xrtoolz.metrics), `Cycle` (pipekit_cycle), `pot_threshold` (xrtoolz.geo). **Name reserved, raises:** `DurationError`. **Proposed:** `persistence` (Report 15 C2), `ClimatologyForecast` (Report 15 C3), `ThresholdEventDetector` (Report 14 E1), `EvaluateByGroup` (Report 14 C1), `split_by` (Report 15 B2), `split_causal` (Report 15 B2), `split_regions` (Report 15 B2), `build_state_catalog` (Report 13 A2).

:::{warning} Differs from the source
The 200 km × 30 day blocks become withheld stations for the mapping variant and a causal split for the forecast variant. The Köppen–Geiger zone split stays as the transfer test; elevation band is a stratification.
:::

(gal-2-2)=
### 2.2 · Precipitation

Precipitation is intermittent, heavy-tailed and the textbook double-penalty variable: a displaced storm is punished twice, so a flat drizzle forecast can beat a correct but shifted one. Point forecasts are close to meaningless for use.

:::{table} Contract 2.2: Precipitation.
:label: tbl-gal-2-2

| Term | Value |
|---|---|
| Task | L1 → L2 mapping, and L3 → L4 forecast |
| Variant | probabilistic |
| Gap the split reproduces | forecasting the future; a climate zone for transfer |
| Splits | causal: held-out years; withhold a region: one climate zone |
| Scores stratified by | season; intensity threshold |
| References, by track | analysis: IMERG Final, MSWEP; observations: GPCC, CPC gauges; radar: MRMS, OPERA |
| Baselines | persistence (1 h); training-period climatology; operational forecast for leads ≥ 6 h |
| Scorers | CRPS; POD, FAR, CSI at 1, 10, 50 mm/h; FSS at several scales; RMSE of log precipitation; distribution of intensities, tails |
| Known failure modes | too much drizzle, too few heavy events; double penalty; convection under-skilled; orography; light rain miscalibrated |
| Contract hash | `82a508619b59…` |
:::

**Stack.** **Today:** `CRPS` (xrtoolz.metrics), `RMSE` (xrtoolz.metrics), `Wasserstein1` (xrtoolz.metrics), `EnsembleCycle` (pipekit_cycle), `SkillByLeadTime` (xrtoolz.metrics). **Name reserved, raises:** `ProbabilityOfDetection`, `FalseAlarmRatio`, `CriticalSuccessIndex`. **Proposed:** `persistence` (Report 15 C2), `ClimatologyForecast` (Report 15 C3), `FSS` (Report 14 B1), `BrierScore` (Report 14 B3), `split_causal` (Report 15 B2), `split_regions` (Report 15 B2), `EvaluateByGroup` (Report 14 C1).

:::{warning} Differs from the source
The source pairs the causal split with 500 km × 14 day blocks. A forecast's test set is the future; blocks assigned at random in time put future data in training, so the contract keeps the causal split alone and scores by season and threshold as stratifications. FSS belongs to the structural lens ([Report 14](reports/14_pipekit_evaluate.md#r14-ws-b)), not the spectral one.
:::

(gal-2-3)=
### 2.3 · Near-surface wind

Wind is a vector, so its direction needs circular statistics and its errors are meaningless in calms. Gusts are the high-impact tail, and wind energy drives the operational stakes.

:::{table} Contract 2.3: Near-surface wind.
:label: tbl-gal-2-3

| Term | Value |
|---|---|
| Task | L1 → L2 mapping, and L3 → L4 forecast |
| Variant | both |
| Gap the split reproduces | mapping at withheld stations; forecasting the future |
| Splits | withhold by `station`: withheld stations; causal: held-out years; withhold a region: flat, coastal, mountainous |
| Scores stratified by | calm against strong wind; terrain class |
| References, by track | reanalysis: ERA5 10 m wind; observations: ASOS, METAR stations; ocean: ASCAT scatterometer; upper air: AMDAR, radiosondes |
| Baselines | persistence; monthly ERA5 climatology (wind roses); operational forecast where licensed |
| Scorers | vector RMSE; circular direction error; CRPS per component, spread-skill; gust, calm and wind-shift events; Weibull parameters; divergence and curl |
| Known failure modes | direction error meaningless in calms; terrain channelling; gusts under-skilled; sea-breeze timing; boundary-layer shear |
| Contract hash | `79ae4f27ff9f…` |
:::

**Stack.** **Today:** `WindDirection` (xrtoolz.atm), `WindSpeed` (xrtoolz.atm), `RMSE` (xrtoolz.metrics), `CRPS` (xrtoolz.metrics), `SpreadSkillRatio` (xrtoolz.metrics), `DivergenceError` (xrtoolz.metrics), `matchup` (geocatalog.matchup), `Cycle` (pipekit_cycle). **Proposed:** `persistence` (Report 15 C2), `ClimatologyForecast` (Report 15 C3), `ThresholdEventDetector` (Report 14 E1), `split_by` (Report 15 B2), `split_causal` (Report 15 B2), `split_regions` (Report 15 B2). **Not proposed yet:** CircularRMSE, Weibull fit.

:::{warning} Differs from the source
Blocks become withheld stations and a causal split. The source names a "Tepper formula" for direction error; it could not be traced, so the contract defines the error as the wrapped angle difference Δθ = atan2(sin(θ̂ − θ), cos(θ̂ − θ)).
:::

(gal-2-4)=
### 2.4 · Mean sea-level pressure

Pressure tracks synoptic systems, cyclones and blocking. It overlaps entry 3.3, which also scores MSLP; the two are kept apart here because this one scores against surface stations and that one against upper-air fields.

:::{table} Contract 2.4: Mean sea-level pressure.
:label: tbl-gal-2-4

| Term | Value |
|---|---|
| Task | L3 → L4, forecast |
| Variant | both |
| Gap the split reproduces | forecasting the future |
| Splits | causal: held-out years, excluding recovery after large eruptions |
| Scores stratified by | ENSO phase; season |
| References, by track | reanalysis: ERA5, JRA-3Q, MERRA-2; observations: GHCN-Daily pressure; upper air: radiosonde heights |
| Baselines | persistence; training-period climatology; operational forecast |
| Scorers | RMSE, bias; CRPS, rank histogram; cyclone tracks (position, intensity, timing); blocking detection; hydrostatic consistency |
| Known failure modes | blocking under-predicted; tropical-cyclone intensity; polar lows; lee cyclones |
| Contract hash | `0612ff1b684f…` |
:::

**Stack.** **Today:** `RMSE` (xrtoolz.metrics), `Bias` (xrtoolz.metrics), `CRPS` (xrtoolz.metrics), `RankHistogram` (xrtoolz.metrics), `InstanceMatcher` (xrtoolz.metrics), `EnsembleCycle` (pipekit_cycle). **Proposed:** `persistence` (Report 15 C2), `ClimatologyForecast` (Report 15 C3), `EvaluateByGroup` (Report 14 C1), `split_causal` (Report 15 B2), `build_state_catalog` (Report 13 A2). **Not proposed yet:** blocking index. **External:** TempestExtremes (cyclone tracking).

:::{warning} Differs from the source
Blocks become a causal split. ENSO phase and season are stratifications of the score. The source's exclusion around large eruptions is kept as a condition on the causal split.
:::


(gal-atmosphere)=
## Atmosphere

The most benchmarked domain (WeatherBench and the evaluations of GraphCast, Pangu-Weather and GenCast). The gallery's distinct contribution is trace gases, which are structured like weather but far less standardized.


(gal-3-1)=
### 3.1 · Trace gases (methane, CO₂, water vapour)

Methane is the headline case for MARS and IMEO: retrieval at L2, gridding at L3 and source attribution at L4 are each a benchmark, and so is the chain. Controlled releases are the only references with a known answer.

:::{table} Contract 3.1: Trace gases (methane, CO₂, water vapour).
:label: tbl-gal-3-1

| Term | Value |
|---|---|
| Task | L0 → L1 → L2 → L4, retrieval to attribution |
| Variant | probabilistic |
| Gap the split reproduces | a region, a source class and a controlled-release campaign unseen in training |
| Splits | withhold a region: one producing basin; withhold by `source_class`: one class; withhold by `campaign`: one controlled-release campaign; causal: held-out years |
| Scores stratified by | emission rate; surface type |
| References, by track | reanalysis: CAMS; columns: TROPOMI, OCO-2/3, EMIT L2; observations: TCCON, NOAA flasks and towers; aircraft: MethaneAIR, CARVE, HIPPO; controlled release: METEC, satellite controlled-release campaigns |
| Baselines | matched filter (plume detection); column retrievals (operational RemoTeC, scientific WFM-DOAS); IME (source rate); CAMS (background) |
| Scorers | column RMSE (ppb); POD by emission rate (1 to 1,000 kg/h); CRPS of emission posteriors; regional totals against inventories; source persistence; mass conservation, positivity |
| Known failure modes | retrieval biases (cloud, aerosol, surface); background mistaken for emission; small plumes detected, mass underestimated; attribution between neighbours; minimum detection limit set by physics |
| Contract hash | `48c16fa63435…` |
:::

**Stack.** **Today:** `MatchedFilter` (geotoolz.matched_filter), `MatchedFilterSNR` (geotoolz.matched_filter), `IMEEstimate` (geotoolz.plume), `CrossSectionalFlux` (geotoolz.plume), `PlumeMask` (geotoolz.plume), `RMSE` (xrtoolz.metrics), `CRPS` (xrtoolz.metrics), `BudgetResidual` (xrtoolz.budgets), `DACycle` (pipekit_cycle). **Name reserved, raises:** `ProbabilityOfDetection`, `FalseAlarmRatio`, `CriticalSuccessIndex`. **Proposed:** `geocatalog.events` (Report 14 E2), `split_regions` (Report 15 B2), `split_by` (Report 15 B2), `split_causal` (Report 15 B2). **Not proposed yet:** cascade contract (stages by hash). **Not pinned:** plumax (plume forward model).

:::{warning} Differs from the source
The source's `BAEMRBaseline` could not be identified as a published retrieval; the column baselines are TROPOMI's operational RemoTeC retrieval and the scientific WFM-DOAS product instead. The source's temporal block becomes a causal split, and its rule that controlled releases be held out in time becomes a split by campaign as well, since the same emitters recur within a campaign. METEC is a ground test facility; the satellite controlled releases are separate campaigns, named as a second reference. The matched filter and the integrated mass enhancement ship in geotoolz today.
:::

(gal-3-2)=
### 3.2 · Three-dimensional wind

Three-dimensional winds carry chemistry, dust and aerosol, and upper-level jets steer extreme weather. Stratospheric and tropical winds are sparse in observations and weakly constrained by balance.

:::{table} Contract 3.2: Three-dimensional wind.
:label: tbl-gal-3-2

| Term | Value |
|---|---|
| Task | L3 → L4, forecast |
| Variant | both |
| Gap the split reproduces | forecasting the future |
| Splits | causal: held-out years |
| Scores stratified by | altitude band; latitude band |
| References, by track | reanalysis: ERA5 (37 pressure levels); observations: radiosondes; ocean surface: ASCAT; motion vectors: GOES, Meteosat, Himawari AMVs; lidar: Aeolus (2018 to 2023) |
| Baselines | persistence; training-period climatology; operational forecast |
| Scorers | vector RMSE per level; CRPS per level, spread-skill; kinetic-energy spectra per level; jet-streak detection; jet position bias; hydrostatic and geostrophic balance |
| Known failure modes | sparse stratosphere; tropical winds without geostrophy; boundary-layer structure; gravity waves smoothed; jet position errors |
| Contract hash | `2bceb7585c55…` |
:::

**Stack.** **Today:** `RMSE` (xrtoolz.metrics), `CRPS` (xrtoolz.metrics), `SpreadSkillRatio` (xrtoolz.metrics), `KESpectralFlux` (xrtoolz.transforms), `PowerSpectrum` (xrtoolz.transforms), `GeostrophicImbalance` (xrtoolz.metrics), `ToPressureLevels` (xrtoolz.interpolate), `Cycle` (pipekit_cycle). **Proposed:** `persistence` (Report 15 C2), `ClimatologyForecast` (Report 15 C3), `ThresholdEventDetector` (Report 14 E1), `EvaluateByGroup` (Report 14 C1), `split_causal` (Report 15 B2), `ProfileSlice` (overview, deferred), `build_state_catalog` (Report 13 A2).

:::{warning} Differs from the source
The 500 km × 14 day blocks become a causal split; altitude band is a stratification of the score, not a split.
:::

(gal-3-3)=
### 3.3 · Geopotential and pressure (Z500, MSLP, tropopause)

The anomaly correlation of 500 hPa geopotential is the headline number of medium-range forecasting; WeatherBench 2 [@rasp2024weatherbench] is the standard benchmark. A contract adds what its documentation cannot: a hash that a submission can be checked against.

:::{table} Contract 3.3: Geopotential and pressure (Z500, MSLP, tropopause).
:label: tbl-gal-3-3

| Term | Value |
|---|---|
| Task | L3 → L4, forecast |
| Variant | both |
| Gap the split reproduces | forecasting the future |
| Splits | causal: held-out years |
| Scores stratified by | ENSO phase; NAO phase; lead time |
| References, by track | reanalysis: ERA5, JRA-3Q, MERRA-2; analysis: ECMWF HRES, GFS analyses; observations: radiosondes |
| Baselines | persistence; training-period climatology; operational forecast; ACC = 0.6 usefulness line |
| Scorers | anomaly correlation (ACC) of Z500; RMSE of Z500; CRPS; blocking detection; hydrostatic and gradient-wind balance; storm-track latitude bias |
| Known failure modes | ACC flatters smooth forecasts; blocking under-predicted; sudden stratospheric warmings; tropical–extratropical interaction |
| Contract hash | `b26e58590340…` |
:::

**Stack.** **Today:** `RemoveClimatology` (xrtoolz.geo), `Correlation` (xrtoolz.metrics), `RMSE` (xrtoolz.metrics), `CRPS` (xrtoolz.metrics), `SkillByLeadTime` (xrtoolz.metrics), `EnsembleCycle` (pipekit_cycle). **Proposed:** `persistence` (Report 15 C2), `ClimatologyForecast` (Report 15 C3), `EvaluateByGroup` (Report 14 C1), `split_causal` (Report 15 B2), `build_state_catalog` (Report 13 A2). **Not proposed yet:** blocking index.

:::{warning} Differs from the source
Blocks become a causal split. ACC is a correlation of anomalies, not a spectral score; it is composed today from `RemoveClimatology` and `Correlation`. The source's "CDC-style no-skill line" is the conventional usefulness threshold ACC = 0.6.
:::


(gal-remote-sensing)=
## Remote sensing

Benchmarks between instruments: less standardized than weather or ocean benchmarks, and essential to every multi-sensor product. Four of the five are not transitions up the L0–L4 ladder (see the box at the top), which is why a contract names its carriers as well as its levels.


(gal-4-1)=
### 4.1 · Multispectral to hyperspectral

Hyperspectral data carry far more information than multispectral data and cover far less of the Earth. Predicting one from the other, or fusing them, would bring hyperspectral detail to multispectral coverage; there is no standard benchmark for it.

:::{table} Contract 4.1: Multispectral to hyperspectral.
:label: tbl-gal-4-1

| Term | Value |
|---|---|
| Task | L1 → L1, between instruments (not a level transition) |
| Variant | deterministic |
| Gap the split reproduces | a scene not seen in training |
| Splits | withhold by `scene_id`: withheld scenes; withhold a region: one land-cover class; causal: a later season |
| Scores stratified by | land-cover class; season |
| References, by track | hyperspectral: PRISMA, EnMAP, EMIT, DESIS; multispectral: Sentinel-2 MSI, Landsat OLI, PlanetScope; airborne: AVIRIS-NG over Sentinel-2 and Landsat scenes |
| Baselines | spectral unmixing (NMF); per-pixel spectral-angle matching; band-ratio regression |
| Scorers | spectral angle, RMSE per band; spectral PSD, smoothness; end-member preservation; downstream task (minerals, water quality); SSIM at multispectral resolution |
| Known failure modes | atmospheric correction differs between instruments; sub-pixel mixing; sun–target–sensor geometry; band centres and widths differ |
| Contract hash | `97b8ef911472…` |
:::

**Stack.** **Today:** `ApplySRF` (geotoolz.radiometry), `SpectralBinning` (geotoolz.spectral), `ContinuumRemoval` (geotoolz.spectral), `RMSE` (xrtoolz.metrics), `PSDScore` (xrtoolz.metrics), `SSIM` (xrtoolz.metrics), `SpatialPatcher` (geopatcher). **Proposed:** `split_by` (Report 15 B2), `split_regions` (Report 15 B2), `split_causal` (Report 15 B2). **Not proposed yet:** spectral angle.

:::{warning} Differs from the source
Leave-one-scene-out is kept as the right unit, and the source's temporal block becomes a causal split (a later season). Band counts per sensor are left out; the contract carries each sensor's spectral response function instead, which `ApplySRF` consumes.
:::

(gal-4-2)=
### 4.2 · Polar-orbiting and geostationary

Polar orbiters see the globe at high resolution a few times a day; geostationary imagers see a hemisphere every few minutes at lower resolution. Nowcasting depends on fusing them.

:::{table} Contract 4.2: Polar-orbiting and geostationary.
:label: tbl-gal-4-2

| Term | Value |
|---|---|
| Task | L1 → L1, between instruments (not a level transition) |
| Variant | deterministic |
| Gap the split reproduces | a sensor pair not seen in training |
| Splits | withhold by `sensor_pair`: withheld pairs; causal: held-out season-years |
| Scores stratified by | latitude band; season; viewing angle |
| References, by track | polar: MODIS, VIIRS, Sentinel-3 OLCI and SLSTR; geostationary: GOES-East and GOES-West, Meteosat, Himawari; observations: coincident overpasses, ground stations |
| Baselines | temporal interpolation to geostationary cadence; bilinear regridding of geostationary data; time-aligned averaging |
| Scorers | RMSE at coincident pixels; spatial PSD; hourly RMSE, diurnal cycle; continuity across the handover; rapid convection detection |
| Known failure modes | viewing-geometry biases; sub-pixel cloud at geostationary resolution; sun-glint geometry; calibration drift; poor geostationary coverage poleward |
| Contract hash | `60d64d9e99eb…` |
:::

**Stack.** **Today:** `matchup` (geocatalog.matchup), `NearestInTime` (geocatalog.matchup), `WithinWindow` (geocatalog.matchup), `ResampleTime` (xrtoolz.interpolate), `RegridLike` (xrtoolz.interpolate), `RMSE` (xrtoolz.metrics), `PSDScore` (xrtoolz.metrics). **Proposed:** `ThresholdEventDetector` (Report 14 E1), `EvaluateByGroup` (Report 14 C1), `split_by` (Report 15 B2), `split_causal` (Report 15 B2).

:::{warning} Differs from the source
The split by sensor pair is kept. Latitude band and season are stratifications, and a causal split by season-year is added so that a model is not scored on the season it was trained on. The geostationary imagers are named by role (GOES-East and GOES-West) rather than by satellite number, which changes with replacements.
:::

(gal-4-3)=
### 4.3 · Radiative transfer emulation

Radiative transfer is the inner loop of every retrieval, and it is slow. An emulator that is fast, accurate and has correct gradients makes differentiable retrievals possible.

:::{table} Contract 4.3: Radiative transfer emulation.
:label: tbl-gal-4-3

| Term | Value |
|---|---|
| Task | state → L1, forward simulation (down the ladder) |
| Variant | deterministic |
| Gap the split reproduces | a physical regime outside the training distribution |
| Splits | withhold by `regime`: cloudy, heavy aerosol, high methane |
| Scores stratified by | surface type; gas concentration range |
| References, by track | reference physics: LBLRTM; operational: MODTRAN, 6S; fast RT models: RTTOV, CRTM; observations: retrievals on real scenes |
| Baselines | look-up table; per-band polynomial regression; RTTOV, CRTM |
| Scorers | spectral RMSE (radiance units); Jacobian fidelity; monotonicity where physics requires; speed-up, forward and backward; out-of-distribution failure rate |
| Known failure modes | background-trained emulators fail on plumes; multiple scattering; accurate radiances with wrong gradients; band edges |
| Contract hash | `125e09b2eb95…` |
:::

**Stack.** **Today:** `SimulationDataset` (pipekit_train), `NeuralForward` (pipekit_cycle), `RMSE` (xrtoolz.metrics). **Proposed:** `split_by` (Report 15 B2). **Not proposed yet:** Jacobian fidelity. **Not pinned:** rtmx (differentiable RTM).

:::{warning} Differs from the source
Regimes outside the training distribution are kept as the test; surface type and gas concentration range become stratifications. RTTOV and CRTM are fast radiative transfer models, not learned emulators. Jacobian fidelity is the score no library ships.
:::

(gal-4-4)=
### 4.4 · Sensor operators (SRF, PSF, noise)

The instrument is the boundary between the world and the data, and its response (spectral response function, point spread function, noise) shapes every product downstream. Correct sensor operators are what observing-system simulation experiments run on.

:::{table} Contract 4.4: Sensor operators (SRF, PSF, noise).
:label: tbl-gal-4-4

| Term | Value |
|---|---|
| Task | truth → L1, forward instrument simulation (down the ladder) |
| Variant | both |
| Gap the split reproduces | a mission phase not seen in training |
| Splits | causal: later mission phase |
| Scores stratified by | band; detector element |
| References, by track | specification: pre-launch SRF, PSF, noise; calibration: lunar, solar, vicarious; observations: matched in-situ and L1 |
| Baselines | convolution with Gaussian noise; vendor sensor model |
| Scorers | radiometric RMSE per band; noise statistics (NEdT, NEdL); MTF reconstruction; positivity; anomaly detection |
| Known failure modes | striping, dead and hot pixels; calibration drift; cross-talk; stray light |
| Contract hash | `3de5ecdef258…` |
:::

**Stack.** **Today:** `ApplySRF` (geotoolz.radiometry), `GaussianNoise` (geotoolz.augment), `DestripeColumn` (geotoolz), `SimulationDataset` (pipekit_train), `RMSE` (xrtoolz.metrics), `PSDScore` (xrtoolz.metrics). **Proposed:** `EvaluateByGroup` (Report 14 C1), `split_causal` (Report 15 B2).

:::{warning} Differs from the source
The split by mission phase is kept; band and detector element are stratifications. The task is a forward simulation from a high-resolution truth to the instrument's L1 data, down the ladder, which is why the contract names its carriers.
:::

(gal-4-5)=
### 4.5 · Multi-satellite fusion

No single sensor gives everything, and most operational L4 products (OSTIA, OC-CCI, IMERG) are fusions inside. A fusion benchmark asks whether combining sensors adds skill, and what happens when one sensor stops.

:::{table} Contract 4.5: Multi-satellite fusion.
:label: tbl-gal-4-5

| Term | Value |
|---|---|
| Task | L1 → L2, gap filling |
| Variant | both |
| Gap the split reproduces | the product with one sensor missing |
| Splits | withhold by `sensor`: one sensor withheld; causal: held-out years |
| Scores stratified by | sensor combination available; season |
| References, by track | analysis: OSTIA, OC-CCI, IMERG, GlobColour; single sensor: where sensors overlap; observations: in situ |
| Baselines | the operational fusion product; inverse-variance weighting; sequential overpass averaging; per-sensor climatology fallback |
| Scorers | RMSE, bias by sensor combination; continuity across handovers and swath edges; cross-sensor bias; events seen only by the fusion; variance preservation |
| Known failure modes | calibration errors propagated; geolocation at high latitude; sun-glint by platform; commissioning and end-of-life drift; best sensor wins, poor fallback |
| Contract hash | `92d04347a6c0…` |
:::

**Stack.** **Today:** `matchup` (geocatalog.matchup), `NearestInTime` (geocatalog.matchup), `SpatialMosaic` (xrtoolz.geo), `RMSE` (xrtoolz.metrics), `Bias` (xrtoolz.metrics), `SpatialPatcher` (geopatcher). **Proposed:** `EvaluateByEvent` (Report 14 C2), `split_by` (Report 15 B2), `split_causal` (Report 15 B2).

:::{warning} Differs from the source
Leave-one-sensor-out is kept: it is the source's split and the right one. The source's split by sensor-combination availability becomes a stratification.
:::


(gal-math)=
## Mathematical models

Benchmarks of a chain of dependent sub-tasks rather than of one transformation. Methane emission estimation is the example: five linked stages from radiative transfer to an annual total.


(gal-5-1)=
### 5.1 · Emission estimation (five-stage chain)

Methane emission estimation is a chain of five dependent sub-tasks, from radiative transfer to an annual total. Each stage is a benchmark, the chain is one too, and errors compound along it, which is what regulatory use has to see.

:::{table} Contract 5.1: Emission estimation (five-stage chain).
:label: tbl-gal-5-1

| Term | Value |
|---|---|
| Task | state → L1 → L2 → L4, a five-stage chain |
| Variant | probabilistic |
| Gap the split reproduces | a region, a source class and a controlled-release campaign unseen in training |
| Splits | withhold a region: one basin (Permian, Bakken, Algeria, Australia); withhold by `source_class`: one class; withhold by `campaign`: one controlled-release campaign; causal: held-out years |
| Scores stratified by | emission rate; stage |
| References, by track | stage 1: LBLRTM, MODTRAN; stage 2: HYSPLIT, FLEXPART, WRF-Chem; METEC; stage 3: METEC, operator campaigns, MethaneAIR; stage 4: long-term monitoring of known emitters; stage 5: EDGAR, national inventories, top-down inversions |
| Baselines | look-up table RTM; Gaussian plume; SNR-thresholded matched filter; persistence of yesterday's detection; tier-1 emission factors |
| Scorers | stage 1: spectral RMSE, Jacobian fidelity; stage 2: concentration RMSE, centroid, IoU; stage 3: POD curve, minimum detection limit; stage 4: Brier score; stage 5: per-source mass error, regional bias, cascade calibration |
| Known failure modes | stage errors compound down the chain; background mistaken for emission; detection floor set by physics; attribution flips; inventory priors regressed to |
| Contract hash | `3594d37b6d8c…` |
:::

**Stack.** **Today:** `SimulationDataset` (pipekit_train), `NeuralForward` (pipekit_cycle), `DACycle` (pipekit_cycle), `Recurrence` (pipekit_cycle), `MatchedFilter` (geotoolz.matched_filter), `DetectionThreshold` (geotoolz.matched_filter), `IMEEstimate` (geotoolz.plume), `WindAdvectionCone` (geotoolz.plume), `RMSE` (xrtoolz.metrics), `InstanceMatcher` (xrtoolz.metrics), `MaskIoU` (xrtoolz.metrics). **Name reserved, raises:** `ProbabilityOfDetection`. **Proposed:** `BrierScore` (Report 14 B3), `geocatalog.events` (Report 14 E2), `CatalogDataset` (Report 11 A1), `split_regions` (Report 15 B2), `split_by` (Report 15 B2), `split_causal` (Report 15 B2). **Not proposed yet:** cascade contract (stages by hash). **Not pinned:** plumax (plume forward model), rtmx (differentiable RTM).

:::{warning} Differs from the source
The source proposes a new `MultiStageEvaluation`. A cascade is several contracts: one per stage and one end to end, each with its own hash, plus a list of how each stage's output feeds the next. That needs one field on the contract, not a second evaluation class.
:::

(gal-patterns)=
## Patterns across the gallery

The source draws five patterns from its entries. Each is recomputed here
from the contracts as data: three hold as stated, two need a qualifier, and a
sixth appears.

- **Three tracks are usually available.** Sixteen of the eighteen entries
  have an observations track; 4.1 uses airborne spectra in its place, and
  5.1 has references per stage instead. The OceanBench pattern generalizes,
  as the source says.
- **Withheld platforms and causal splits dominate.** Thirteen entries
  withhold something by a column (a mission, platform, station, float,
  cruise, scene, sensor, sensor pair, regime, source class or campaign),
  sixteen need a causal split, and eight withhold a named region. None needs random space-time blocks. The source
  saw the first half of this ("the hardest splits are platform-based"); the
  second half is Report 15's correction.
- **The variant follows the variable.** Five entries require a probabilistic
  contract (salinity, biogeochemistry, precipitation, trace gases and the
  emission chain), eight accept both, and five are deterministic. The source's
  point stands: the contract declares the variant, and the runner refuses a
  mismatch (Report 15's D1).
- **Events are everywhere.** Fifteen of the eighteen entries score at least
  one event (eddies, heatwaves, blooms, plumes, cyclones, blocking, rapid
  convection). The three that do not (4.1, 4.3, 4.4) are instrument and
  emulator benchmarks. Pieces exist (instance matching and mask IoU in
  xrtoolz, the Okubo–Weiss field, external trackers), but no entry can
  produce its contingency scores today: POD, FAR and CSI are reserved names
  and the detectors are proposed ([Report 14](reports/14_pipekit_evaluate.md#r14-ws-e)).
- **Cascades need their own pattern, but not a new class.** Two entries are
  chains (3.1, retrieval to attribution, and 5.1). [](#gal-5-1) shows the
  pattern on 5.1, as stage contracts linked by hash; 3.1 would split the same
  way into its L2, L3 and L4 stages.
- **Every entry stratifies.** All eighteen break their scores down by
  something (region, season, phase, altitude, water type, sensor
  combination). Report 14's grouping wrapper, `EvaluateByGroup` (C1), is
  needed by eight entries, as many as the region split, and every other
  stratification goes through `EvaluateByRegion`, which ships.

(gal-needs)=
## What the stack needs

The gallery needs no new package. Most of what it cites ships today; the
rest is proposed in Reports 11 to 15 or deferred in the overview, and six
pieces are proposed nowhere.

:::{figure} images/benchmarks_gallery/readiness.svg
:label: fig-gallery-readiness
:alt: A grid with one row per gallery entry and one dot per stack piece the entry needs, coloured by status. Ocean, land and atmosphere entries have between four and ten pieces available today. Radiative transfer emulation needs the fewest pieces. The emission chain needs the most pieces overall, with eleven available today.

Readiness of the eighteen contracts, from the notebook: one dot per stack
piece an entry needs. The contract and the multi-reference runner, needed by
every entry, are included.
:::

:::{table} The pieces the gallery needs that do not run today, by how many entries need them.
:label: tbl-gal-needs

| Piece | Entries | Home | Status |
|---|---|---|---|
| `BenchmarkContract`, `Benchmark(references=…)` | 18 | pipekit-evaluate | proposed, [Report 15](reports/15_benchmarking.md#r15-ws-a) A1 and D2 |
| `split_causal`, `split_by` | 16, 13 | geocatalog | proposed, Report 15 B2 |
| `ClimatologyForecast` | 11 | xrtoolz | proposed, Report 15 C3 |
| `persistence` | 9 | xrtoolz | proposed, Report 15 C2 |
| `split_regions` | 8 | geocatalog | proposed, Report 15 B2 |
| `EvaluateByGroup` | 8 | xrtoolz | proposed, [Report 14](reports/14_pipekit_evaluate.md#r14-ws-c) C1 |
| `ThresholdEventDetector`, `MarineHeatwaveDetector` | 6, 1 | xrtoolz | proposed, Report 14 E1 |
| `build_state_catalog` | 5 | geocatalog | proposed, [Report 13](reports/13_state_catalog.md#r13-ws-a) A2 |
| `ProfileSlice` | 5 | geocatalog | deferred in the [overview](overview.md#sec-plan) until an in-situ project needs it |
| `optimal_interpolation` | 4 | xrtoolz | proposed, Report 15 C1 |
| `FSS`, `BrierScore`, `geocatalog.events` | 2 each | xrtoolz, geocatalog | proposed, Report 14 B1, B3, E2 |
| `EvaluateByEvent`, `CatalogDataset` | 1 each | xrtoolz, pipekit-train | proposed, Report 14 C2, [Report 11](reports/11_pipekit_train.md#r11-ws-a) A1 |
| `ProbabilityOfDetection`, `FalseAlarmRatio`, `CriticalSuccessIndex` | 6, 4, 2 | xrtoolz | names reserved; Report 14 B2 implements them |
| `DurationError` | 2 | xrtoolz | name reserved; no report proposes an implementation yet |
| blocking index | 2 | xrtoolz, beside the event detectors | not proposed yet |
| cascade contract (stages by hash) | 2 | pipekit-evaluate, one field on `BenchmarkContract` | not proposed yet |
| circular direction error | 1 | xrtoolz.metrics, beside `xrtoolz.atm.WindDirection` | not proposed yet |
| Weibull parameter recovery | 1 | xrtoolz.metrics (distributional) | not proposed yet |
| spectral angle | 1 | geotoolz.spectral | not proposed yet |
| Jacobian fidelity | 1 | a scorer over pipekit-jax gradients | not proposed yet |
:::

The six pieces nobody proposes are small: each is a function over arrays
the stack already holds. The cascade contract is the one with design
content, and it is a field, not a class:

```{code-block} python
:caption: The emission chain of entry 5.1 as linked contracts (sketch).
:linenos:

# Shapes: five stage contracts and one end-to-end contract, each hashed
stages = [rtm, plume, pod, persistence, total]       # BenchmarkContracts  proposed (A1)
chain = pe.BenchmarkContract(
    name="methane-emission-chain", task="state->L1->L2->L4",
    stages=[s.content_hash() for s in stages],       # gap: a new field
    feeds={"plume": "rtm", "pod": "plume", "persistence": "pod",
           "total": ["pod", "persistence"]},          # arguments sketched
    scorers=[...],                                    # end to end
)
# a submission scores each stage against its own contract and the chain
# against `chain`; the report records all six hashes
```

The source's `MultiStageEvaluation` would compute "cascade uncertainty".
Here that is a scorer on the end-to-end contract (does the predicted
uncertainty of the total match its empirical error?), so no second runner is
needed.

(gal-not)=
## What this gallery is not

- **Not a leaderboard.** Running submissions is the work of the projects
  that host one; the gallery gives them contracts.
- **Not a standard.** OceanBench, WeatherBench and the others own their
  domains; these entries show how such a benchmark would read as a contract.
- **Not complete.** Sea ice, terrestrial carbon fluxes and fire radiative
  power, among others, are missing.
- **Not final.** Each contract needs a domain expert's review of its metrics,
  baselines and splits before it is pre-registered, and its hash will change
  when it is.

(gal-references)=
## References

### Sources

- The source draft, "Benchmarks Gallery", the companion to the Report 15
  scoping report (supplied by the author).
- OceanBench [@johnson2023oceanbench] and WeatherBench 2
  [@rasp2024weatherbench], the community benchmarks entries 1.1 and 3.3 are
  modelled on.
- Block cross-validation for structured data [@roberts2017], the background
  to the split rule.
- The pinned code in `references/` and the gallery's notebook,
  `notebooks/reports/benchmarks_gallery.py`.
