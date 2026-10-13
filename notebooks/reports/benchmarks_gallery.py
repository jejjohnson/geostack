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
# # Benchmarks gallery, checked
#
# Holds the eighteen draft contracts of `docs/benchmarks_gallery.md` as data
# and checks them against the pinned code:
#
# 1. every stack name an entry marks `today` imports from the module given;
# 2. every name marked `reserved` imports and raises `NotImplementedError`;
# 3. every name marked `proposed` with a target module does not exist there yet,
#    and every split an entry declares has the stack piece that makes it;
# 4. each contract gets a content hash (`pipekit.hashing`), as Report 15's
#    workstream A would give it;
# 5. the counts the page quotes about the contracts (tracks, variants, split
#    kinds, needs, reserved names) are computed here and asserted.
#
# It also draws the page's readiness figure into
# `docs/images/benchmarks_gallery/readiness.svg`. Pinned revisions: pipekit
# 0.0.2 at `ef3c17f`, xrtoolz 0.0.5 at `308c7f2`, geotoolz 0.10.2 (geocatalog
# 0.5.1) at `088475e`. It needs only those libraries; it runs in seconds.

# %%
from __future__ import annotations

import importlib
import importlib.util
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipekit.hashing import sha256_hex, stable_json


# %% [markdown]
# ## The vocabulary
#
# A stack reference is `(name, status, where)`. `today`: `where` is the module
# that must export `name`. `reserved`: exported, but construction raises.
# `proposed`: `where` names the report and workstream, and `module` (when
# given) is where the name must not exist yet. `external`: a tool outside the
# stack. `unpinned`: a stack library not in `references/`, so not checked.
# `gap`: needed, and proposed by no report; the gallery's own finding.

# %%
@dataclass(frozen=True)
class Ref:
    name: str
    status: str  # today | reserved | proposed | external | unpinned | gap
    where: str
    module: str | None = None  # for proposed: where it must not exist yet


@dataclass
class Entry:
    id: str
    name: str
    domain: str
    task: str  # level transition, Report 15's A1
    variant: str  # deterministic | probabilistic | both
    gap: str  # the deployment gap the split reproduces (Report 15, stage 2)
    splits: list[dict]
    stratify: list[str]  # score decompositions, not splits
    tracks: dict[str, str]
    baselines: list[str]
    scorers: list[str]
    failures: list[str]
    refs: list[Ref] = field(default_factory=list)

    def contract(self) -> dict:
        d = asdict(self)
        d.pop("refs")
        return d

    def content_hash(self) -> str:
        # h = SHA-256 of the key-sorted JSON (Report 15, eq. r15-eq-hash)
        return sha256_hex(stable_json(self.contract()))


T = "today"
R15, R14, R13, R11 = "Report 15", "Report 14", "Report 13", "Report 11"
# shared references
OI = Ref("optimal_interpolation", "proposed", f"{R15} C1", "xrtoolz.interpolate")
PERSIST = Ref("persistence", "proposed", f"{R15} C2", "xrtoolz.geo")
CLIM = Ref("ClimatologyForecast", "proposed", f"{R15} C3", "xrtoolz.geo")
CALCCLIM = Ref("CalculateClimatology", T, "xrtoolz.geo")
CONTRACT = Ref("BenchmarkContract", "proposed", f"{R15} A1", "pipekit_evaluate")
TRACKS = Ref("Benchmark(references=…)", "proposed", f"{R15} D2")
SPLIT_BY = Ref("split_by", "proposed", f"{R15} B2", "geocatalog")
SPLIT_CAUSAL = Ref("split_causal", "proposed", f"{R15} B2", "geocatalog")
SPLIT_REGIONS = Ref("split_regions", "proposed", f"{R15} B2", "geocatalog")
SPLIT_BLOCKS = Ref("split_blocks", "proposed", f"{R15} B1", "geocatalog")
STATES = Ref("build_state_catalog", "proposed", f"{R13} A2", "geocatalog.build")
EVENTS = Ref("geocatalog.events", "proposed", f"{R14} E2")
DETECT = Ref("ThresholdEventDetector", "proposed", f"{R14} E1", "xrtoolz")
MHW = Ref("MarineHeatwaveDetector", "proposed", f"{R14} E1", "xrtoolz")
FSS = Ref("FSS", "proposed", f"{R14} B1", "xrtoolz.metrics")
BRIER = Ref("BrierScore", "proposed", f"{R14} B3", "xrtoolz.metrics")
ENERGY = Ref("EnergyScore", "proposed", f"{R14} B3", "xrtoolz.metrics")
NSE = Ref("NSE", "proposed", f"{R14} B3", "xrtoolz.metrics")
BYGROUP = Ref("EvaluateByGroup", "proposed", f"{R14} C1", "xrtoolz.metrics")
BYEVENT = Ref("EvaluateByEvent", "proposed", f"{R14} C2", "xrtoolz.metrics")
CATDS = Ref("CatalogDataset", "proposed", f"{R11} A1", "pipekit_train")
POD = Ref("ProbabilityOfDetection", "reserved", "xrtoolz.metrics.object")
FAR = Ref("FalseAlarmRatio", "reserved", "xrtoolz.metrics.object")
CSI = Ref("CriticalSuccessIndex", "reserved", "xrtoolz.metrics.object")
DUR = Ref("DurationError", "reserved", "xrtoolz.metrics.object")
PROFILES = Ref("ProfileSlice", "proposed", "overview, deferred", "geocatalog")
RMSE = Ref("RMSE", T, "xrtoolz.metrics")
BIAS = Ref("Bias", T, "xrtoolz.metrics")
CRPS = Ref("CRPS", T, "xrtoolz.metrics")
SSR = Ref("SpreadSkillRatio", T, "xrtoolz.metrics")
RANK = Ref("RankHistogram", T, "xrtoolz.metrics")
REL = Ref("ReliabilityCurve", T, "xrtoolz.metrics")
PSD = Ref("PSDScore", T, "xrtoolz.metrics")
SSIM = Ref("SSIM", T, "xrtoolz.metrics")
CORR = Ref("Correlation", T, "xrtoolz.metrics")
BYLEAD = Ref("SkillByLeadTime", T, "xrtoolz.metrics")
BYREGION = Ref("EvaluateByRegion", T, "xrtoolz.metrics")
W1 = Ref("Wasserstein1", T, "xrtoolz.metrics")
MATCH = Ref("matchup", T, "geocatalog.matchup")
NEAREST = Ref("NearestInTime", T, "geocatalog.matchup")
INSTANCE = Ref("InstanceMatcher", T, "xrtoolz.metrics")
CYCLE = Ref("Cycle", T, "pipekit_cycle")
ENSCYCLE = Ref("EnsembleCycle", T, "pipekit_cycle")
DACYCLE = Ref("DACycle", T, "pipekit_cycle")
SIMDS = Ref("SimulationDataset", T, "pipekit_train")
NEURAL = Ref("NeuralForward", T, "pipekit_cycle")
REMCLIM = Ref("RemoveClimatology", T, "xrtoolz.geo")
CASCADE = Ref("cascade contract (stages by hash)", "gap", "this gallery")

# %% [markdown]
# ## The eighteen entries

# %%
E: list[Entry] = []

# ---------------------------------------------------------------- 1 · Ocean
E.append(Entry(
    "1.1", "Sea surface height", "Ocean", "L1->L2", "both",
    "mapping where one altimetry mission flew, from the others",
    [{"kind": "by", "column": "mission", "test": "one nadir mission"},
     {"kind": "causal", "test": "held-out year"}],
    ["region (western boundary currents, coastal)"],
    {"reanalysis": "GLORYS12", "analysis": "DUACS L4",
     "observations": "withheld along-track mission; SWOT where available"},
    ["OI (L, τ fitted to the training years)", "persistence (forecast variant)",
     "training-period climatology"],
    ["RMSE of SSH and of geostrophic velocity", "PSD score, resolved scale",
     "geostrophic balance error", "eddy detection (POD, FAR)",
     "CRPS, spread-skill (ensemble variant)"],
    ["double penalty rewards smoothed maps", "spectra too steep at small scales",
     "western boundary currents under-resolved", "coastal cells worst",
     "SWOT cross-track artefacts learned as signal"],
    [OI, PERSIST, CLIM, RMSE, PSD, Ref("resolved_scale", T, "xrtoolz.metrics"),
     Ref("GeostrophicBalanceError", T, "xrtoolz.metrics"),
     Ref("GeostrophicVelocities", T, "xrtoolz.ocn"),
     Ref("CalculateSSHAlongtrack", T, "xrtoolz.ocn"),
     Ref("along_track_psd_score", T, "xrtoolz.metrics"),
     Ref("OkuboWeiss", T, "xrtoolz.ocn"), Ref("py-eddy-tracker", "external", "eddy tracking"),
     POD, FAR, CRPS, SSR, SPLIT_BY, SPLIT_CAUSAL, STATES, TRACKS, CONTRACT],
))
E.append(Entry(
    "1.2", "Sea surface temperature", "Ocean", "L1->L2", "deterministic",
    "filling cloud gaps: real cloud masks from other days laid over clear scenes",
    [{"kind": "mask transplant", "source": "cloud masks of other days"},
     {"kind": "by", "column": "platform", "test": "withheld drifters"},
     {"kind": "causal", "test": "held-out year"}],
    ["time of day", "front regions"],
    {"analysis": "OSTIA, ESA CCI SST, MUR, OISST",
     "observations": "drifters, near-surface Argo, ship lines"},
    ["OI (L, τ fitted to the training years)", "training-period climatology",
     "persistence with climatology fill"],
    ["RMSE, bias", "PSD score", "SSIM, FSS", "marine heatwave detection",
     "gradient magnitude near fronts"],
    ["clear-sky bias in training", "diurnal aliasing of polar orbiters",
     "fronts smoothed", "tidal coastal variability missed"],
    [OI, CLIM, PERSIST, RMSE, BIAS, PSD, SSIM, FSS, MHW, POD, DUR,
     Ref("GradientDifference", T, "xrtoolz.metrics"),
     Ref("SimulatedClouds", T, "geotoolz.augment"), SPLIT_BY, SPLIT_CAUSAL,
     PROFILES, TRACKS, CONTRACT],
))
E.append(Entry(
    "1.3", "Sea surface salinity", "Ocean", "L1->L2", "probabilistic",
    "mapping where a held-out Argo float measured; a river system for transfer",
    [{"kind": "by", "column": "float_id", "test": "withheld floats"},
     {"kind": "regions", "test": "one major river system"}],
    ["region (river plumes, polar)"],
    {"analysis": "CCI SSS, CMEMS L4 SSS, ISAS",
     "observations": "near-surface Argo salinity, thermosalinograph lines"},
    ["World Ocean Atlas climatology", "OI (L, τ fitted to the training years)",
     "persistence, weekly"],
    ["RMSE, bias", "CRPS", "river-plume detection", "T–S consistency",
     "regional means against ISAS"],
    ["satellite retrieval errors spatially structured",
     "near-surface Argo gradient unresolved", "river-plume edges sub-daily",
     "ice contamination at high latitude"],
    [CLIM, OI, PERSIST, RMSE, BIAS, CRPS, DETECT,
     Ref("density_from_ts", T, "xrtoolz.ocn"), BYREGION, SPLIT_BY, SPLIT_REGIONS,
     PROFILES, Ref("pyrox-gp", "unpinned", "Gaussian-process gap filling"),
     TRACKS, CONTRACT],
))
E.append(Entry(
    "1.4", "Ocean colour", "Ocean", "L1->L2", "deterministic",
    "a biogeochemical province not seen in training; in-situ by cruise",
    [{"kind": "regions", "test": "one Longhurst province"},
     {"kind": "by", "column": "cruise", "test": "withheld cruises"},
     {"kind": "causal", "test": "held-out years"}],
    ["water type (open ocean, coastal)", "season"],
    {"analysis": "OC-CCI, GlobColour",
     "observations": "HPLC chlorophyll (SeaBASS, NOMAD)",
     "operational": "NASA OC4, GIOP"},
    ["NASA OC4 band ratio", "GIOP", "monthly OC-CCI climatology"],
    ["RMSE of log10 chlorophyll, bias by water type", "PSD score",
     "bloom onset (POD, FAR with timing tolerance)",
     "distribution distance of log chlorophyll", "front alignment with SST"],
    ["atmospheric correction in coastal waters", "sun glint", "cloud shadows",
     "sediment-rich water overestimated; models learn the bias"],
    [Ref("BandRatio", T, "geotoolz.spectral"), CLIM, RMSE, BIAS, PSD, W1, POD,
     FAR, BYGROUP, SPLIT_REGIONS, SPLIT_BY, SPLIT_CAUSAL, PROFILES, TRACKS,
     CONTRACT],
))
E.append(Entry(
    "1.5", "Ocean biogeochemistry", "Ocean", "L0->L2", "probabilistic",
    "a float's whole record, unseen in training",
    [{"kind": "by", "column": "float_id", "test": "withheld floats"},
     {"kind": "causal", "test": "held-out years"}],
    ["basin", "depth"],
    {"climatology": "WOA", "analysis": "CMEMS BGC reanalysis",
     "observations": "BGC-Argo floats, GLODAPv2 bottles"},
    ["WOA monthly climatology", "OI (slow scales)", "ESPER"],
    ["RMSE per variable and depth", "CRPS per depth, calibration on bottles",
     "stoichiometric consistency", "oxygen-minimum-zone and DCM detection"],
    ["sensor calibration drift sets the floor", "near-surface optics, thermocline",
     "very low oxygen overestimated", "sparse floats south of 60° S"],
    [CLIM, OI, Ref("ESPER", "external", "empirical seawater property estimates"),
     RMSE, CRPS, REL, DETECT, Ref("DensityInversionFraction", T, "xrtoolz.metrics"),
     SPLIT_BY, SPLIT_CAUSAL, PROFILES,
     Ref("pyrox-gp", "unpinned", "3-D Gaussian-process gap filling"), TRACKS,
     CONTRACT],
))

# ----------------------------------------------------------------- 2 · Land
E.append(Entry(
    "2.1", "Near-surface temperature", "Land", "L1->L2 and L3->L4", "both",
    "mapping at withheld stations; forecasting the future",
    [{"kind": "by", "column": "station", "test": "withheld stations"},
     {"kind": "causal", "test": "held-out years"},
     {"kind": "regions", "test": "one Köppen–Geiger zone"}],
    ["elevation band", "climate zone"],
    {"reanalysis": "ERA5-Land, MERRA-2", "observations": "GHCN-Daily stations",
     "satellite": "MODIS, VIIRS, SLSTR land surface temperature"},
    ["persistence (24 h, 168 h)", "training-period climatology",
     "operational forecast where licensed"],
    ["RMSE, MAE, bias", "CRPS, rank histogram", "diurnal phase and amplitude",
     "heatwave detection and duration", "bias against elevation"],
    ["urban heat islands (absent from ERA5-Land too)", "lapse-rate bias in terrain",
     "uneven station network", "diurnal phase over deserts and snow",
     "skin against 2 m temperature"],
    [PERSIST, CLIM, RMSE, Ref("MAE", T, "xrtoolz.metrics"), BIAS, CRPS, RANK,
     DETECT, DUR, BYGROUP, CYCLE, Ref("pot_threshold", T, "xrtoolz.geo"),
     SPLIT_BY, SPLIT_CAUSAL, SPLIT_REGIONS, STATES, TRACKS, CONTRACT],
))
E.append(Entry(
    "2.2", "Precipitation", "Land", "L1->L2 and L3->L4", "probabilistic",
    "forecasting the future; a climate zone for transfer",
    [{"kind": "causal", "test": "held-out years"},
     {"kind": "regions", "test": "one climate zone"}],
    ["season", "intensity threshold"],
    {"analysis": "IMERG Final, MSWEP", "observations": "GPCC, CPC gauges",
     "radar": "MRMS, OPERA"},
    ["persistence (1 h)", "training-period climatology",
     "operational forecast for leads ≥ 6 h"],
    ["CRPS", "POD, FAR, CSI at 1, 10, 50 mm/h", "FSS at several scales",
     "RMSE of log precipitation", "distribution of intensities, tails"],
    ["too much drizzle, too few heavy events", "double penalty",
     "convection under-skilled", "orography", "light rain miscalibrated"],
    [PERSIST, CLIM, CRPS, POD, FAR, CSI, FSS, RMSE, W1, BRIER, ENSCYCLE, BYLEAD,
     SPLIT_CAUSAL, SPLIT_REGIONS, BYGROUP, TRACKS, CONTRACT],
))
E.append(Entry(
    "2.3", "Near-surface wind", "Land", "L1->L2 and L3->L4", "both",
    "mapping at withheld stations; forecasting the future",
    [{"kind": "by", "column": "station", "test": "withheld stations"},
     {"kind": "causal", "test": "held-out years"},
     {"kind": "regions", "test": "flat, coastal, mountainous"}],
    ["calm against strong wind", "terrain class"],
    {"reanalysis": "ERA5 10 m wind", "observations": "ASOS, METAR stations",
     "ocean": "ASCAT scatterometer", "upper air": "AMDAR, radiosondes"},
    ["persistence", "monthly ERA5 climatology (wind roses)",
     "operational forecast where licensed"],
    ["vector RMSE", "circular direction error", "CRPS per component, spread-skill",
     "gust, calm and wind-shift events", "Weibull parameters",
     "divergence and curl"],
    ["direction error meaningless in calms", "terrain channelling",
     "gusts under-skilled", "sea-breeze timing", "boundary-layer shear"],
    [PERSIST, CLIM, Ref("WindDirection", T, "xrtoolz.atm"),
     Ref("WindSpeed", T, "xrtoolz.atm"), RMSE, CRPS, SSR, DETECT,
     Ref("CircularRMSE", "gap", "this gallery"),
     Ref("Weibull fit", "gap", "this gallery"),
     Ref("DivergenceError", T, "xrtoolz.metrics"), MATCH, CYCLE, SPLIT_BY,
     SPLIT_CAUSAL, SPLIT_REGIONS, TRACKS, CONTRACT],
))
E.append(Entry(
    "2.4", "Mean sea-level pressure", "Land", "L3->L4", "both",
    "forecasting the future",
    [{"kind": "causal", "test": "held-out years",
      "exclude": "recovery after large eruptions"}],
    ["ENSO phase", "season"],
    {"reanalysis": "ERA5, JRA-3Q, MERRA-2", "observations": "GHCN-Daily pressure",
     "upper air": "radiosonde heights"},
    ["persistence", "training-period climatology", "operational forecast"],
    ["RMSE, bias", "CRPS, rank histogram", "cyclone tracks (position, intensity, timing)",
     "blocking detection", "hydrostatic consistency"],
    ["blocking under-predicted", "tropical-cyclone intensity", "polar lows",
     "lee cyclones"],
    [PERSIST, CLIM, RMSE, BIAS, CRPS, RANK, INSTANCE,
     Ref("TempestExtremes", "external", "cyclone tracking"),
     Ref("blocking index", "gap", "this gallery"), ENSCYCLE, BYGROUP,
     SPLIT_CAUSAL, STATES, TRACKS, CONTRACT],
))

# ----------------------------------------------------------- 3 · Atmosphere
E.append(Entry(
    "3.1", "Trace gases (methane, CO₂, water vapour)", "Atmosphere",
    "L0->L1->L2->L4", "probabilistic",
    "a region, a source class and a controlled-release campaign unseen in training",
    [{"kind": "regions", "test": "one producing basin"},
     {"kind": "by", "column": "source_class", "test": "one class"},
     {"kind": "by", "column": "campaign", "test": "one controlled-release campaign"},
     {"kind": "causal", "test": "held-out years"}],
    ["emission rate", "surface type"],
    {"reanalysis": "CAMS", "columns": "TROPOMI, OCO-2/3, EMIT L2",
     "observations": "TCCON, NOAA flasks and towers",
     "aircraft": "MethaneAIR, CARVE, HIPPO",
     "controlled release": "METEC, satellite controlled-release campaigns"},
    ["matched filter (plume detection)",
     "column retrievals (operational RemoTeC, scientific WFM-DOAS)",
     "IME (source rate)", "CAMS (background)"],
    ["column RMSE (ppb)", "POD by emission rate (1 to 1,000 kg/h)",
     "CRPS of emission posteriors", "regional totals against inventories",
     "source persistence", "mass conservation, positivity"],
    ["retrieval biases (cloud, aerosol, surface)", "background mistaken for emission",
     "small plumes detected, mass underestimated", "attribution between neighbours",
     "minimum detection limit set by physics"],
    [Ref("MatchedFilter", T, "geotoolz.matched_filter"),
     Ref("MatchedFilterSNR", T, "geotoolz.matched_filter"),
     Ref("IMEEstimate", T, "geotoolz.plume"),
     Ref("CrossSectionalFlux", T, "geotoolz.plume"),
     Ref("PlumeMask", T, "geotoolz.plume"), RMSE, CRPS, POD, FAR, CSI,
     Ref("BudgetResidual", T, "xrtoolz.budgets"), DACYCLE, EVENTS,
     Ref("plumax", "unpinned", "plume forward model"), SPLIT_REGIONS, SPLIT_BY,
     SPLIT_CAUSAL, CASCADE, TRACKS, CONTRACT],
))
E.append(Entry(
    "3.2", "Three-dimensional wind", "Atmosphere", "L3->L4", "both",
    "forecasting the future",
    [{"kind": "causal", "test": "held-out years"}],
    ["altitude band", "latitude band"],
    {"reanalysis": "ERA5 (37 pressure levels)", "observations": "radiosondes",
     "ocean surface": "ASCAT", "motion vectors": "GOES, Meteosat, Himawari AMVs",
     "lidar": "Aeolus (2018 to 2023)"},
    ["persistence", "training-period climatology", "operational forecast"],
    ["vector RMSE per level", "CRPS per level, spread-skill",
     "kinetic-energy spectra per level", "jet-streak detection",
     "jet position bias", "hydrostatic and geostrophic balance"],
    ["sparse stratosphere", "tropical winds without geostrophy",
     "boundary-layer structure", "gravity waves smoothed", "jet position errors"],
    [PERSIST, CLIM, RMSE, CRPS, SSR, Ref("KESpectralFlux", T, "xrtoolz.transforms"),
     Ref("PowerSpectrum", T, "xrtoolz.transforms"),
     Ref("GeostrophicImbalance", T, "xrtoolz.metrics"),
     Ref("ToPressureLevels", T, "xrtoolz.interpolate"), DETECT, BYGROUP, CYCLE,
     SPLIT_CAUSAL, PROFILES, STATES, TRACKS, CONTRACT],
))
E.append(Entry(
    "3.3", "Geopotential and pressure (Z500, MSLP, tropopause)", "Atmosphere",
    "L3->L4", "both",
    "forecasting the future",
    [{"kind": "causal", "test": "held-out years"}],
    ["ENSO phase", "NAO phase", "lead time"],
    {"reanalysis": "ERA5, JRA-3Q, MERRA-2",
     "analysis": "ECMWF HRES, GFS analyses", "observations": "radiosondes"},
    ["persistence", "training-period climatology", "operational forecast",
     "ACC = 0.6 usefulness line"],
    ["anomaly correlation (ACC) of Z500", "RMSE of Z500", "CRPS",
     "blocking detection", "hydrostatic and gradient-wind balance",
     "storm-track latitude bias"],
    ["ACC flatters smooth forecasts", "blocking under-predicted",
     "sudden stratospheric warmings", "tropical–extratropical interaction"],
    [PERSIST, CLIM, REMCLIM, CORR, RMSE, CRPS, BYLEAD,
     Ref("blocking index", "gap", "this gallery"), ENSCYCLE, BYGROUP,
     SPLIT_CAUSAL, STATES, TRACKS, CONTRACT],
))

# ------------------------------------------------------- 4 · Remote sensing
E.append(Entry(
    "4.1", "Multispectral to hyperspectral", "Remote sensing", "L1->L1",
    "deterministic",
    "a scene not seen in training",
    [{"kind": "by", "column": "scene_id", "test": "withheld scenes"},
     {"kind": "regions", "test": "one land-cover class"},
     {"kind": "causal", "test": "a later season"}],
    ["land-cover class", "season"],
    {"hyperspectral": "PRISMA, EnMAP, EMIT, DESIS",
     "multispectral": "Sentinel-2 MSI, Landsat OLI, PlanetScope",
     "airborne": "AVIRIS-NG over Sentinel-2 and Landsat scenes"},
    ["spectral unmixing (NMF)", "per-pixel spectral-angle matching",
     "band-ratio regression"],
    ["spectral angle, RMSE per band", "spectral PSD, smoothness",
     "end-member preservation", "downstream task (minerals, water quality)",
     "SSIM at multispectral resolution"],
    ["atmospheric correction differs between instruments", "sub-pixel mixing",
     "sun–target–sensor geometry", "band centres and widths differ"],
    [Ref("ApplySRF", T, "geotoolz.radiometry"),
     Ref("SpectralBinning", T, "geotoolz.spectral"),
     Ref("ContinuumRemoval", T, "geotoolz.spectral"),
     Ref("spectral angle", "gap", "this gallery"), RMSE, PSD, SSIM, SPLIT_BY,
     SPLIT_REGIONS, SPLIT_CAUSAL, Ref("SpatialPatcher", T, "geopatcher"), TRACKS, CONTRACT],
))
E.append(Entry(
    "4.2", "Polar-orbiting and geostationary", "Remote sensing", "L1->L1",
    "deterministic",
    "a sensor pair not seen in training",
    [{"kind": "by", "column": "sensor_pair", "test": "withheld pairs"},
     {"kind": "causal", "test": "held-out season-years"}],
    ["latitude band", "season", "viewing angle"],
    {"polar": "MODIS, VIIRS, Sentinel-3 OLCI and SLSTR",
     "geostationary": "GOES-East and GOES-West, Meteosat, Himawari",
     "observations": "coincident overpasses, ground stations"},
    ["temporal interpolation to geostationary cadence",
     "bilinear regridding of geostationary data", "time-aligned averaging"],
    ["RMSE at coincident pixels", "spatial PSD", "hourly RMSE, diurnal cycle",
     "continuity across the handover", "rapid convection detection"],
    ["viewing-geometry biases", "sub-pixel cloud at geostationary resolution",
     "sun-glint geometry", "calibration drift", "poor geostationary coverage poleward"],
    [MATCH, NEAREST, Ref("WithinWindow", T, "geocatalog.matchup"),
     Ref("ResampleTime", T, "xrtoolz.interpolate"),
     Ref("RegridLike", T, "xrtoolz.interpolate"), RMSE, PSD, DETECT, BYGROUP,
     SPLIT_BY, SPLIT_CAUSAL, TRACKS, CONTRACT],
))
E.append(Entry(
    "4.3", "Radiative transfer emulation", "Remote sensing", "state->L1",
    "deterministic",
    "a physical regime outside the training distribution",
    [{"kind": "by", "column": "regime", "test": "cloudy, heavy aerosol, high methane"}],
    ["surface type", "gas concentration range"],
    {"reference physics": "LBLRTM", "operational": "MODTRAN, 6S",
     "fast RT models": "RTTOV, CRTM", "observations": "retrievals on real scenes"},
    ["look-up table", "per-band polynomial regression", "RTTOV, CRTM"],
    ["spectral RMSE (radiance units)", "Jacobian fidelity",
     "monotonicity where physics requires", "speed-up, forward and backward",
     "out-of-distribution failure rate"],
    ["background-trained emulators fail on plumes", "multiple scattering",
     "accurate radiances with wrong gradients", "band edges"],
    [SIMDS, NEURAL, RMSE, Ref("Jacobian fidelity", "gap", "this gallery"),
     Ref("rtmx", "unpinned", "differentiable RTM"), SPLIT_BY, TRACKS, CONTRACT],
))
E.append(Entry(
    "4.4", "Sensor operators (SRF, PSF, noise)", "Remote sensing", "truth->L1",
    "both",
    "a mission phase not seen in training",
    [{"kind": "causal", "test": "later mission phase"}],
    ["band", "detector element"],
    {"specification": "pre-launch SRF, PSF, noise",
     "calibration": "lunar, solar, vicarious",
     "observations": "matched in-situ and L1"},
    ["convolution with Gaussian noise", "vendor sensor model"],
    ["radiometric RMSE per band", "noise statistics (NEdT, NEdL)",
     "MTF reconstruction", "positivity", "anomaly detection"],
    ["striping, dead and hot pixels", "calibration drift", "cross-talk",
     "stray light"],
    [Ref("ApplySRF", T, "geotoolz.radiometry"),
     Ref("GaussianNoise", T, "geotoolz.augment"),
     Ref("DestripeColumn", T, "geotoolz"), SIMDS, RMSE, PSD, BYGROUP,
     SPLIT_CAUSAL, TRACKS, CONTRACT],
))
E.append(Entry(
    "4.5", "Multi-satellite fusion", "Remote sensing", "L1->L2", "both",
    "the product with one sensor missing",
    [{"kind": "by", "column": "sensor", "test": "one sensor withheld"},
     {"kind": "causal", "test": "held-out years"}],
    ["sensor combination available", "season"],
    {"analysis": "OSTIA, OC-CCI, IMERG, GlobColour",
     "single sensor": "where sensors overlap", "observations": "in situ"},
    ["the operational fusion product", "inverse-variance weighting",
     "sequential overpass averaging", "per-sensor climatology fallback"],
    ["RMSE, bias by sensor combination", "continuity across handovers and swath edges",
     "cross-sensor bias", "events seen only by the fusion", "variance preservation"],
    ["calibration errors propagated", "geolocation at high latitude",
     "sun-glint by platform", "commissioning and end-of-life drift",
     "best sensor wins, poor fallback"],
    [MATCH, NEAREST, Ref("SpatialMosaic", T, "xrtoolz.geo"), RMSE, BIAS, BYEVENT,
     Ref("SpatialPatcher", T, "geopatcher"), SPLIT_BY, SPLIT_CAUSAL, TRACKS, CONTRACT],
))

# --------------------------------------------------- 5 · Mathematical models
E.append(Entry(
    "5.1", "Emission estimation (five-stage chain)", "Mathematical models",
    "state->L1->L2->L4", "probabilistic",
    "a region, a source class and a controlled-release campaign unseen in training",
    [{"kind": "regions", "test": "one basin (Permian, Bakken, Algeria, Australia)"},
     {"kind": "by", "column": "source_class", "test": "one class"},
     {"kind": "by", "column": "campaign", "test": "one controlled-release campaign"},
     {"kind": "causal", "test": "held-out years"}],
    ["emission rate", "stage"],
    {"stage 1": "LBLRTM, MODTRAN", "stage 2": "HYSPLIT, FLEXPART, WRF-Chem; METEC",
     "stage 3": "METEC, operator campaigns, MethaneAIR",
     "stage 4": "long-term monitoring of known emitters",
     "stage 5": "EDGAR, national inventories, top-down inversions"},
    ["look-up table RTM", "Gaussian plume", "SNR-thresholded matched filter",
     "persistence of yesterday's detection", "tier-1 emission factors"],
    ["stage 1: spectral RMSE, Jacobian fidelity", "stage 2: concentration RMSE, centroid, IoU",
     "stage 3: POD curve, minimum detection limit", "stage 4: Brier score",
     "stage 5: per-source mass error, regional bias, cascade calibration"],
    ["stage errors compound down the chain", "background mistaken for emission",
     "detection floor set by physics", "attribution flips",
     "inventory priors regressed to"],
    [CASCADE, SIMDS, NEURAL, DACYCLE, Ref("Recurrence", T, "pipekit_cycle"),
     Ref("MatchedFilter", T, "geotoolz.matched_filter"),
     Ref("DetectionThreshold", T, "geotoolz.matched_filter"),
     Ref("IMEEstimate", T, "geotoolz.plume"),
     Ref("WindAdvectionCone", T, "geotoolz.plume"), RMSE, POD, BRIER, INSTANCE,
     Ref("MaskIoU", T, "xrtoolz.metrics"), EVENTS, CATDS,
     Ref("plumax", "unpinned", "plume forward model"),
     Ref("rtmx", "unpinned", "differentiable RTM"), SPLIT_REGIONS, SPLIT_BY,
     SPLIT_CAUSAL, TRACKS, CONTRACT],
))
len(E), Counter(e.domain for e in E)

# %% [markdown]
# ## Check every reference against the pinned code

# %%
def exists(module: str, name: str) -> bool:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False
    return hasattr(mod, name)


def check(ref: Ref) -> str:
    if ref.status == "today":
        return "ok" if exists(ref.where, ref.name) else "MISSING"
    if ref.status == "reserved":
        if not exists(ref.where, ref.name):
            return "MISSING"
        try:
            getattr(importlib.import_module(ref.where), ref.name)()
        except NotImplementedError:
            return "ok"
        return "IMPLEMENTED"
    if ref.status == "proposed" and ref.module:
        return "EXISTS" if exists(ref.module, ref.name) else "ok"
    return "ok"


KIND_REF = {"by": "split_by", "causal": "split_causal", "regions": "split_regions"}
for e in E:  # every split an entry declares has the stack piece that makes it
    for sp in e.splits:
        if sp["kind"] in KIND_REF:
            assert KIND_REF[sp["kind"]] in {r.name for r in e.refs}, (e.id, sp)
refs = {r for e in E for r in e.refs}
bad = {r.name: check(r) for r in refs if check(r) != "ok"}
by_status = Counter(r.status for r in refs)
bad, by_status

# %%
assert bad == {}
assert len(E) == 18 and len(refs) == by_status.total()

# %% [markdown]
# ## Hashes and the counts the page quotes

# %%
hashes = {e.id: e.content_hash()[:12] for e in E}
tracks_obs = sum(any(k in e.tracks for k in ("observations",)) for e in E)
variants = Counter(e.variant for e in E)
split_kinds = Counter(s["kind"] for e in E for s in e.splits)
entries_by_kind = Counter(k for e in E for k in {s["kind"] for s in e.splits})
causal_only = [e.id for e in E if [s["kind"] for s in e.splits] == ["causal"]]
needs = Counter(r.name for e in E for r in e.refs if r.status in ("proposed", "gap"))
gaps = sorted({r.name for e in E for r in e.refs if r.status == "gap"})
per_entry = {e.id: Counter(r.status for r in e.refs) for e in E}
hashes, tracks_obs, variants, entries_by_kind, causal_only, needs.most_common(12), gaps

# %%
assert hashes == {
    "1.1": "867c6d7f7bc2", "1.2": "2c6b783b5d20", "1.3": "0e47b9ba77fd",
    "1.4": "40840af4f114", "1.5": "8fb617842097", "2.1": "cbe972a8f817",
    "2.2": "82a508619b59", "2.3": "79ae4f27ff9f", "2.4": "0612ff1b684f",
    "3.1": "48c16fa63435", "3.2": "2bceb7585c55", "3.3": "b26e58590340",
    "4.1": "97b8ef911472", "4.2": "60d64d9e99eb", "4.3": "125e09b2eb95",
    "4.4": "3de5ecdef258", "4.5": "92d04347a6c0", "5.1": "3594d37b6d8c",
}
assert by_status == Counter({"today": 61, "proposed": 18, "gap": 6, "reserved": 4,
                             "unpinned": 4, "external": 3})
assert tracks_obs == 16
assert variants == Counter({"both": 8, "probabilistic": 5, "deterministic": 5})
assert entries_by_kind == Counter({"causal": 16, "by": 13, "regions": 8,
                                   "mask transplant": 1})
assert causal_only == ["2.4", "3.2", "3.3", "4.4"]
assert dict(needs) == {
    "BenchmarkContract": 18, "Benchmark(references=…)": 18, "split_causal": 16,
    "split_by": 13, "ClimatologyForecast": 11, "persistence": 9,
    "split_regions": 8, "EvaluateByGroup": 8, "ThresholdEventDetector": 6,
    "build_state_catalog": 5, "ProfileSlice": 5, "optimal_interpolation": 4,
    "FSS": 2, "BrierScore": 2, "blocking index": 2, "geocatalog.events": 2,
    "cascade contract (stages by hash)": 2, "MarineHeatwaveDetector": 1,
    "CircularRMSE": 1, "Weibull fit": 1, "spectral angle": 1,
    "Jacobian fidelity": 1, "EvaluateByEvent": 1, "CatalogDataset": 1,
}
reserved_entries = [e.id for e in E if any(r.status == "reserved" for r in e.refs)]
reserved_uses = Counter(r.name for e in E for r in e.refs if r.status == "reserved")
assert reserved_entries == ["1.1", "1.2", "1.4", "2.1", "2.2", "3.1", "5.1"]
assert reserved_uses == Counter({"ProbabilityOfDetection": 6, "FalseAlarmRatio": 4,
                                 "CriticalSuccessIndex": 2, "DurationError": 2})
assert gaps == ["CircularRMSE", "Jacobian fidelity", "Weibull fit", "blocking index",
                "cascade contract (stages by hash)", "spectral angle"]

# %%
EVENT = {"ProbabilityOfDetection", "ThresholdEventDetector", "MarineHeatwaveDetector",
         "InstanceMatcher", "blocking index", "py-eddy-tracker", "TempestExtremes",
         "EvaluateByEvent"}
no_obs = [e.id for e in E if "observations" not in e.tracks]
with_events = [e.id for e in E if any(r.name in EVENT for r in e.refs)]
cascades = [e.id for e in E if any(r.name.startswith("cascade") for r in e.refs)]
stratified = sum(bool(e.stratify) for e in E)
no_obs, with_events, cascades, stratified

# %%
assert no_obs == ["4.1", "5.1"]
assert len(with_events) == 15
assert set(e.id for e in E) - set(with_events) == {"4.1", "4.3", "4.4"}
assert cascades == ["3.1", "5.1"] and stratified == 18

# %% [markdown]
# ## The readiness figure
#
# One row per entry, one cell per stack reference, coloured by status: what
# runs today, what a report proposes, and what nobody has proposed yet.

# %%
root = Path.cwd()
while not (root / "myst.yml").exists() and root != root.parent:
    root = root.parent
spec = importlib.util.spec_from_file_location(
    "overview_figures", root / "docs/images/overview/make_figures.py")
ov = importlib.util.module_from_spec(spec)
sys.modules["overview_figures"] = ov
spec.loader.exec_module(ov)

ORDER = ["today", "reserved", "proposed", "gap", "external", "unpinned"]
# solid squares for the stack's own statuses, outlines for everything outside it
HUE = {"today": (ov.STATE[0], ov.STATE[0]), "reserved": (ov.OBS[0], ov.OBS[1]),
       "proposed": (ov.ENG[0], ov.ENG[0]), "gap": (ov.ML[0], ov.ML[0]),
       "external": (ov.LINE, ov.PANEL), "unpinned": (ov.LINE, "#e2e8f0")}
LABEL = {"today": "today", "reserved": "name reserved", "proposed": "proposed",
         "gap": "not proposed yet", "external": "external tool",
         "unpinned": "library not pinned"}
cell, gapx, x0, y0 = 15, 3, 380, 56
width = x0 + max(len(e.refs) for e in E) * (cell + gapx) + 24
fig = ov.Svg(
    int(width), 120 + len(E) * (cell + 6),
    "Readiness of the eighteen gallery contracts. One row per entry, one square"
    " per stack piece the entry needs, coloured by status: today (it imports"
    " from the pinned code), name reserved, proposed by a report, not proposed"
    " yet, an external tool, or a stack library not pinned here.",
)
x = 24.0
for s in ORDER:
    stroke, fill = HUE[s]
    fig.rect(x, 18, cell, cell, stroke=stroke, fill=fill)
    fig.text(x + cell + 6, 30, LABEL[s], size=12, color=ov.INK, anchor="start")
    x += cell + 16 + 7.2 * len(LABEL[s])
for row, e in enumerate(E):
    y = y0 + row * (cell + 6)
    fig.text(24, y + 12, f"{e.id}  {e.name}"[:60], size=12, color=ov.INK,
             anchor="start")
    for col, r in enumerate(sorted(e.refs, key=lambda r: ORDER.index(r.status))):
        stroke, fill = HUE[r.status]
        fig.rect(x0 + col * (cell + gapx), y, cell, cell, stroke=stroke, fill=fill)
fig.h = int(y0 + len(E) * (cell + 6) + 12)
out = root / "docs/images/benchmarks_gallery"
out.mkdir(parents=True, exist_ok=True)
(out / "readiness.svg").write_text(fig.render(), encoding="utf-8")
{e.id: dict(per_entry[e.id]) for e in E}
