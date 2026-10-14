---
title: The geo data stack
subtitle: Report 16 · Realignment plan for pipekit, xrtoolz and geotoolz across 18 distributions
short_title: 16 · Geo data stack
description: How pipekit, the xrtoolz family and the five geotoolz packages should divide the work. Checked against the revisions pinned in references/ after geotoolz grew geotoolz-cloud and geotoolz-products; one synthetic methane scene runs through every lane, and four of the six seams that need fixing are exercised as they stand.
keywords:
  - pipekit
  - geotoolz
  - geotoolz-catalog
  - geotoolz-patcher
  - geotoolz-cloud
  - geotoolz-products
  - xrtoolz
  - package ownership
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

Three repositories ship 18 distributions: pipekit (7), xrtoolz (6) and
geotoolz (5). None of them merges into another. pipekit is the composition
layer; geotoolz-cloud, geotoolz-catalog and geotoolz-patcher serve both
carrier lanes; geotoolz and geotoolz-products are the GeoTensor lane; the
xrtoolz family is the xarray lane. One synthetic methane scene runs through
the packages that touch it (pipekit and pipekit-array, geocatalog,
geopatcher, geotoolz and the xrtoolz core, reader and top-level package) and
the matched filter's noise lands within 3.2 % of its closed form. Six seams
still need work: two `Signature` classes that cannot share a pipeline, eight
model wrappers for three backends, shared xarray vocabulary that lives in a
downloader, a training dataset that imports a geopatcher module removed on
8 October, no rule for where a mission reader lives, and a distribution name
on PyPI that belongs to another project. The running example exercises four
of them on the pinned code (the signatures, the vocabulary's dependency, the
training dataset and the dependency names); the model wrappers and the
mission readers are read from the source in the audit, and pipekit-jax,
geotoolz-products, xrtoolz-einx, xrtoolz-grad and xrtoolz-sklearn are not
imported by the run. This page decides each one and pulls together
the open questions Reports 3, 6, 7 and 11 left on them.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Proposal. It replaces the 7 October draft of this plan, which predates geotoolz-cloud and geotoolz-products; this page is the source of truth |
| **Reading time** | about 30 minutes |
| **Audience** | anyone changing where a symbol lives in pipekit, xrtoolz or geotoolz, or publishing them |
| **Companions** | [Report 2](02_pipekit_surface.md) (pipekit core); [Report 3](03_sister_libraries.md) (geotoolz and xrtoolz on pipekit, whose open questions 4 and 7 this page answers); [Report 6](06_geocatalog.md) and [Report 7](07_geopatcher.md) (whose PyPI-name questions this page answers); [Report 11](11_pipekit_train.md) (whose A5 is this page's D1) |
| **Code checked against** | the revisions pinned in `references/`: pipekit and its six sister packages at 0.0.2 (`ef3c17f`); geotoolz 0.10.2 with geotoolz-catalog 0.5.1, geotoolz-patcher 0.10.2, geotoolz-cloud 0.4.1 and geotoolz-products 0.4.2 (`088475e`); xrtoolz 0.0.5 with xrtoolz-core 0.0.2 and xrtoolz-reader 0.0.4 (`308c7f2`). Read on 14 October 2026 |
| **Running example** | every output below is from one run of `notebooks/reports/report16_geo_data_stack.py`, which seeds every draw and asserts each number the page shows |
:::

:::{important} What changed since the 7 October draft
geotoolz split twice after the first draft. Object storage left the patcher
for **geotoolz-cloud** (geotoolz #431), and the sensor readers left geotoolz
for **geotoolz-products** (#420). The same refactor renamed
`geopatcher.time` to `geopatcher.temporal`, which broke pipekit-train's
window dataset ([](#r16-stage-6)). Two findings of the draft are fixed on
`main`: the two `divide_evenly` helpers now have distinct names
(`exact_quotient` and `count_steps`, #434), and the obstore pool has one
home in geotoolz-cloud. The draft also said xrtoolz imports `BBox` and
`TimeRange` from xrreader; it imports `Variable`, `REGISTRY`,
`apply_cf_attrs` and the subset functions ([](#r16-ws-c)).
:::

(r16-summary)=
## Summary

**Decision.** No package merges into another, and the lanes stay apart: no
GeoTensor package imports an xarray one or the reverse. Each overlap below
gets one owner, every move but one leaves a deprecated alias for the 0.x
series (the window dataset's old import, which does not work today), and
the stack publishes under names nobody else holds.

- **A. pipekit owns the composition contract.** pipekit and xrtoolz-core each
  define a `Signature`, with different layouts, so a pipekit-array operator
  after an xrtoolz one fails at `summary` time. Keep pipekit's, give it
  xrtoolz-core's mapping view, and make public the three helpers geotoolz and
  geotoolz-patcher import from `pipekit._base`.
- **B. One model-operator family.** Eight classes wrap models for three
  backends, and two unrelated ones are named `JaxModelOp`. Each lane keeps
  its carrier marshalling; each backend is wrapped once. Eight become five,
  with aliases.
- **C. xrtoolz-core owns the shared xarray vocabulary.** The CF variable
  registry, the validators and the subset functions live in xrtoolz-reader,
  so a base install of xrtoolz pulls the downloaders. Move them into
  xrtoolz-core; xrtoolz-reader keeps its archives.
- **D. geotoolz-patcher owns windows for every carrier.** pipekit-train's
  `XarrayWindowDataset` imports `geopatcher.time`, which no longer exists,
  and it is the only edge from pipekit into the geo stack. Fix the import
  (Report 11's A5), then move the dataset into the patcher behind a `[train]`
  extra; the xarray lane drops `xrpatcher` from its design.
- **E. A mission reader lives with its output carrier.** geotoolz-products and
  xrtoolz-reader both read mission products, with no rule between them. A
  reader that returns a georeferenced raster goes to geotoolz-products; one
  that returns a swath, 2-D latitude and longitude, profiles or stations goes
  to xrtoolz-reader.
- **F. Packaging and release.** The name `pipekit` on PyPI belongs to another
  project, the two workspace roots pin pipekit two incompatible ways, and two
  faults shipped because only geotoolz runs a packaging test. Publish as
  `pipekit-core`, pin by version range, copy the packaging test, and release
  in tier order.

**End state.** pipekit-core holds every carrier-agnostic contract and imports
nothing from the geo stack. geotoolz-cloud moves bytes, geotoolz-catalog says
which data, geotoolz-patcher says which piece, geotoolz computes on
GeoTensors, geotoolz-products decodes gridded missions, and the xrtoolz
family computes on xarray over xrtoolz-core's vocabulary. All 18
distributions install from PyPI into one environment.

(r16-ownership)=
## Ownership rules

Foundations sit at the top and every arrow points up: a package imports only
from tiers above it, and the two lanes never import each other.

:::{figure} ../images/reports/r16_ownership.svg
:label: fig-r16-ownership
:alt: Tier 1: pipekit-core, and the external georeader, xarray and obstore. Tier 2: the pipekit subpackages, geotoolz-cloud and xrtoolz-core. Tier 3, shared by both lanes: geotoolz-catalog and geotoolz-patcher, with the proposed integrations.train module. Tier 4: the GeoTensor lane (geotoolz, geotoolz-products) and the xarray lane (xrtoolz, its einx, grad and sklearn bridges, and xrtoolz-reader), which never import each other. Scripts at the bottom.

The geo data stack after the realignment. Solid arrows are base
dependencies, dashed ones optional extras; the dashed blue box is the one new
surface.
:::

1. **pipekit-core** (`import pipekit`) owns `Operator`, `Sequential`, `Graph`,
   `Signature`, serialisation, control flow, the parallel maps, observation,
   QC and the shared protocols. It has no dependencies and never imports
   xarray, georeader or a geo package.
2. **pipekit-array, -cycle, -evaluate, -experiment, -jax and -train** own
   orchestration on that contract: Array API operators, cycles, benchmarks,
   tracking, the one Equinox model operator, and training loops. None imports
   a geo package (today pipekit-train does; [](#r16-ws-d)).
3. **geotoolz-cloud** (`import geocloud`) owns object storage: the
   process-wide obstore pool, URI verbs, credentials, local copies, COG and
   HDF reads and writes, and the fsspec view. It imports georeader, obstore
   and rasterio, and never pipekit or another geo package.
4. **geotoolz-catalog** (`import geocatalog`) owns *which data*: `GeoSlice`,
   catalogs and their backends, discovery sources with footprints (STAC, CMR,
   earthaccess, GEE), matchup, staging and `field_for`. It serves both lanes,
   and reaches geocloud and geopatcher only through its `[cloud]` and
   `[patch]` extras.
5. **geotoolz-patcher** (`import geopatcher`) owns *which piece* for every
   carrier: split and merge, temporal stencils, the patch cache, and the
   integrations that put patching into a pipekit graph or, after D, a
   pipekit-train dataset. It imports pipekit and geocloud only inside those
   integrations and fields.
6. **geotoolz-products** (`import geoproducts`) owns decoding missions whose
   output is a georeferenced raster: GOES-R ABI, Himawari AHI, Carbon Mapper,
   the toy sensor. Its presets bind geotoolz operators to channel names and
   import geotoolz only inside the `[operators]` extra.
7. **geotoolz** owns operators on `GeoTensor`: reads of one source,
   radiometry, indices, masks, matched filters, plume quantification, and
   GeoTensor marshalling for models. It imports pipekit, and geopatcher and
   geocloud only through extras; it never imports xarray-lane packages.
8. **xrtoolz-core** (`import xrcore`) owns the xarray `Operator` and, after C,
   the shared xarray vocabulary: the CF variable registry, validation and
   subsetting. It imports pipekit, xarray and NumPy, and after C pandas,
   which the moved subsetting code uses and which it declares directly
   (xarray already requires it, so the install does not grow).
9. **xrtoolz-einx, xrtoolz-grad and xrtoolz-sklearn** own labelled einsum,
   finite differences and the one xarray scikit-learn bridge. They import
   xrtoolz-core at most.
10. **xrtoolz-reader** (`import xrreader`) owns request-subset archives
    (CMEMS, CDS, AEMET) and mission readers whose output is a swath, 2-D
    latitude and longitude, profiles or stations. After C it imports the
    vocabulary from xrtoolz-core.
11. **xrtoolz** owns diagnostics, metrics and ocean and atmosphere physics on
    xarray. It imports the xrtoolz family and pipekit, and xrtoolz-reader only
    through a `[reader]` extra after C.
12. **The lanes meet only in scripts.** A script that needs both lanes passes
    arrays between them; radiometry and spectral indices stay in geotoolz,
    and xrtoolz does not grow a second copy.

:::{table} The dependency tiers after the realignment.
:label: tbl-r16-tiers

| Tier | Packages | May import | Role |
|---|---|---|---|
| 1 Foundations | pipekit-core; georeader, xarray, obstore (external) | nothing in the stack | contracts, carriers, bytes |
| 2 Framework and storage | pipekit-array, -cycle, -evaluate, -experiment, -jax, -train; geotoolz-cloud; xrtoolz-core | tier 1 | orchestration; object storage; the xarray operator and vocabulary |
| 3 Shared geo | geotoolz-catalog, geotoolz-patcher | tiers 1–2 (pipekit and geocloud through extras) | which data, which piece |
| 4 Lanes | geotoolz, geotoolz-products · xrtoolz, -einx, -grad, -sklearn, -reader | tiers 1–3, never the other lane | compute and decode on one carrier |
| 5 Entry points | geostack scripts and notebooks | everything | compose both lanes |
:::

(r16-running)=
## The running example: one methane scene through both lanes

**The problem.** A 256 × 256 scene of 30 m pixels in UTM zone 13N, observed
in $c = 40$ SWIR bands between 2100 and 2450 nm on 15 July 2026, contains a
methane plume. The goal is the emission rate $Q$ in kg s⁻¹. The scene is
synthetic so every stage has an exact answer to check against.

**The data.** Let $x_p \in \mathbb{R}^c$ be the radiance spectrum of pixel
$p$, $\mu \in \mathbb{R}^c$ the background mean, $\Sigma \in \mathbb{R}^{c \times c}$
the background covariance (rank 6 plus a diagonal), $t \in \mathbb{R}^c$ the
unit-absorption target (the radiance change per ppm·m of methane, built from
two absorption bands near 2300 and 2360 nm) and $\alpha_p$ the enhancement in
ppm·m, peaking at 1500 ppm·m. Each pixel follows the linear model

$$
x_p = \mu + L z_p + \alpha_p t , \qquad z_p \sim \mathcal{N}(0, I_c) , \qquad L L^\top = \Sigma .
$$ (r16-eq-model)

The best a matched filter can do is set by the noise floor
$\sigma_\alpha = (t^\top \Sigma^{-1} t)^{-1/2}$, which is 103.9 ppm·m here.

(r16-stage-1)=
### Stage 1 · Which data (geotoolz-catalog → geotoolz-patcher)

**TL;DR.** Index the scene file, ask for the day's slice over the scene, and
get back one field the patcher can split.

**Problem.** A catalog row holds a footprint, a time interval and a path. A
`GeoSlice` $s = (b, I, r, \mathrm{crs})$ is a request: bounds $b$, a time
interval $I$, a resolution $r$ and a CRS. `field_for` returns the rows that
intersect $s$, mosaicked onto its grid, as one `RasterField`.

```{code-block} python
:caption: Catalog to field; nothing in this stage changes under the plan.
:linenos:

# Shapes: c = 40 bands, H = W = 256 pixels
from geocatalog import GeoSlice, build, patch

cat = build.build_raster_catalog([path], filename_regex=r"scene_(?P<date>\d{8})",
                                 date_format="%Y%m%d")            # 1 row        today
# s = (b, I, r, crs): the scene's bounds, 15 July, 30 m, UTM 13N
slice_ = GeoSlice(bounds=(x0, y0 - H * res, x0 + W * res, y0),
                  interval=pd.Interval(day_start, day_end, closed="both"),
                  resolution=(30.0, 30.0), crs="EPSG:32613")     # today
field = patch.field_for(cat, slice_)          # rows ∩ s → RasterField (c, H, W)   today
# → RasterField, domain shape (40, 256, 256)
```

This is the seam [Report 6](06_geocatalog.md) documents: the catalog
imports geopatcher only inside `geocatalog.patch`, through the `[patch]`
extra, which is why rule 4 holds today.

(r16-stage-2)=
### Stage 2 · Which piece, and the enhancement (geotoolz-patcher → geotoolz)

**TL;DR.** Cut the scene into overlapping chips, run a matched filter fitted
on each chip, and blend the chips back into one enhancement map.

**Problem.** Chips are $n = 128$ pixels on a side at stride $s = 64$, so the
scene of $N_x = 256$ pixels has

$$
K = \left\lfloor \frac{N_x - n}{s} \right\rfloor + 1 = 3
$$ (r16-eq-k)

anchors per axis and $K^2 = 9$ chips. On chip $k$ the filter fits a median
$\hat{\mu}_k$ and a Ledoit–Wolf shrinkage covariance $\hat{\Sigma}_k$
[@ledoit2004], then estimates the enhancement [@thompson2015]

$$
\hat{\alpha}_k(x) = \frac{(x - \hat{\mu}_k)^\top \hat{\Sigma}_k^{-1} t}{t^\top \hat{\Sigma}_k^{-1} t} .
$$ (r16-eq-mf)

Each chip carries a separable Hann weight $w_k(p) = h(i)\,h(j)$, with
$h(i) = \sin^2\!\big(\pi (i + \tfrac{1}{2}) / n\big)$ sampled at pixel
centres, and the merged map is the weight-normalised sum over the chips
$\mathcal{K}(p)$ that cover pixel $p$:

$$
\hat{\alpha}(p) = \frac{\sum_{k \in \mathcal{K}(p)} w_k(p)\, \hat{\alpha}_k(p)}{\sum_{k \in \mathcal{K}(p)} w_k(p)} .
$$ (r16-eq-ola)

```{code-block} python
:caption: Split, filter and merge as one pipekit graph.
:linenos:

# Shapes: c = 40, n = 128 chip side, K² = 9 chips
from geopatcher import SpatialPatcher, spatial
from geotoolz import matched_filter as mf
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches
from pipekit import Sequential

patcher = SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(128, 128)),
    sampler=spatial.sampler.RegularStride(step=(64, 64)),
    window=spatial.window.Hann(),                    # h(i) = sin²(π (i + ½) / n)
    aggregation=spatial.aggregation.OverlapAdd(),
)                                                                    # today
pipe = Sequential([
    GridSampler(patcher=patcher),                    # (c, H, W) → 9 × (c, n, n)      today
    # α̂ₖ(x) = (x − μ̂ₖ)ᵀ Σ̂ₖ⁻¹ t / (tᵀ Σ̂ₖ⁻¹ t)
    ApplyToChips(operator=mf.MatchedFilter(target=t, fit_on_call=True)),
                                                     # 9 × (c, n, n) → 9 × (n, n)    today
    # α̂(p) = ∑ₖ wₖ(p) α̂ₖ(p) / ∑ₖ wₖ(p)
    MergePatches(aggregation=spatial.aggregation.OverlapAdd(),
                 domain=field.domain),               # 9 × (n, n) → (H, W)            today
])
alpha = pipe(field)                                  # GeoTensor (H, W), ppm·m         today
# → background std 107.2 ppm·m against σ_α = 103.9 (+3.2 %); peak 1588 against 1500
# → same CRS and transform as the scene; correlation with the truth 0.73
```

The same pipeline with an identity operator in place of the filter returns
the scene with a maximum absolute error of 0.0, so the split and merge add
nothing of their own. This stage is the payoff of rule 5: the patcher is
carrier-agnostic, the filter is pure GeoTensor, and pipekit's `Sequential`
joins them.

(r16-stage-3)=
### Stage 3 · Mass and mask (geotoolz)

**TL;DR.** Turn the enhancement into column mass and keep the pixels that
stand clear of the noise.

**Problem.** With $M_{\mathrm{CH_4}} = 0.01604$ kg mol⁻¹ the molar mass of
methane and $V_m = 0.024465$ m³ mol⁻¹ the molar volume of air at 298.15 K
and 1 atm, the column mass is

$$
\Delta\Omega_p = 10^{-6}\, \hat{\alpha}_p\, \frac{M_{\mathrm{CH_4}}}{V_m} \quad [\mathrm{kg\,m^{-2}}] .
$$ (r16-eq-mass)

The plume mask $\mathcal{P}$ keeps connected regions of at least 50 pixels
where $\Delta\Omega_p > 3 \hat{\sigma}$, and $\hat{\sigma}$ is the robust
noise estimate $1.4826 \operatorname{median}_p |\Delta\Omega_p - \operatorname{median} \Delta\Omega|$.

```{code-block} python
:caption: Column mass and a 3σ̂ mask.
:linenos:

# Shapes: H = W = 256
from geotoolz import plume

# ΔΩ = 10⁻⁶ α̂ M_CH₄ / V_m
kg_m2 = plume.ColumnToMass(gas="CH4", units_in="ppm_m")(alpha)   # (H, W) → (H, W)  today
# σ̂ = 1.4826 · median |ΔΩ − median ΔΩ|
sigma_kg = 1.4826 * np.median(np.abs(kg_m2 - np.median(kg_m2)))  # scalar     user
mask = plume.PlumeMask(threshold=3 * sigma_kg, min_area_px=50)(kg_m2)  # (H, W) bool  today
# → σ̂ = 111.2 ppm·m equivalent; 1246 plume pixels, 3 outside the true 3σ_α region
```

`ColumnToMass` matches the closed form {eq}`r16-eq-mass` to 1e-12. Stage 3
uses only geotoolz, so nothing in it moves.

(r16-stage-4)=
### Stage 4 · Wind (xrtoolz → xrtoolz-core)

**TL;DR.** Take the 10 m wind from a reanalysis cube at the overpass and
turn it into an effective wind for the plume.

**Problem.** $u_{10}$ and $v_{10}$ are the 10 m wind components on an
ERA5-shaped grid of 24 hourly steps and 5 × 5 points, with
$u_{10} = 3 + 0.5 \sin(2\pi h / 24)$ m s⁻¹ and $v_{10} = 4$ m s⁻¹. The wind
speed and the effective wind of Varon et al. [@varon2018] are

$$
U_{10} = \sqrt{u_{10}^2 + v_{10}^2} , \qquad U_{\mathrm{eff}} = 0.33\, U_{10} + 0.45 .
$$ (r16-eq-wind)

```{code-block} python
:caption: Wind speed on xarray, interpolated to the overpass.
:linenos:

# Shapes: T = 24 hourly steps, 5 × 5 grid
from xrtoolz.atm import WindSpeed

# U₁₀ = √(u₁₀² + v₁₀²)
ws = WindSpeed(u="u10", v="v10", name="ws10")(era5)["ws10"]       # (T, 5, 5) → (T, 5, 5)  today
U10 = float(ws.interp(time=overpass, latitude=39.5, longitude=-105.5))  # → scalar  today
# U_eff = 0.33 U₁₀ + 0.45
U_eff = 0.33 * U10 + 0.45                                          # scalar     user
# → U₁₀ = 4.722 m s⁻¹, U_eff = 2.008 m s⁻¹
```

`WindSpeed` writes CF attributes through `apply_cf_attrs` and the `Variable`
registry, which it imports from xrtoolz-reader today. After C the same names
come from xrtoolz-core and nothing in this block changes. A real run would
download the cube with `xrreader.CDSSource`; it is built in memory here so
the page needs no credentials.

(r16-stage-5)=
### Stage 5 · Emission rate (geotoolz)

**TL;DR.** Sum the mass inside the mask and divide by the time the wind takes
to carry it the plume's length.

**Problem.** With $A_p = 900$ m² the area of pixel $p$ and $\ell$ the plume
length (the mask's longest axis), the integrated mass enhancement, the
emission rate and its 1σ uncertainty at geotoolz's default fraction are

$$
\mathrm{IME} = \sum_{p \in \mathcal{P}} \Delta\Omega_p A_p , \qquad
Q = \frac{U_{\mathrm{eff}}\, \mathrm{IME}}{\ell} , \qquad \sigma_Q = 0.5\, Q .
$$ (r16-eq-ime)

```{code-block} python
:caption: The emission rate from the GeoTensor and xarray stages together.
:linenos:

# Shapes: H = W = 256; U_eff scalar from stage 4
# IME = ∑_{p∈𝒫} ΔΩ_p A_p,  Q = U_eff IME / ℓ
est = plume.IMEEstimate(wind_speed=U_eff)(kg_m2, mask)   # (H, W), (H, W) → dict   today
# → IME 534.9 kg, ℓ 1651.1 m, Q 0.651 kg s⁻¹, σ_Q 0.325 kg s⁻¹
```

The script passes one float from the xarray lane to the GeoTensor lane;
nothing imports across, which is rule 12. Within the mask the true IME is
584.6 kg, so the estimate is 8.5 % low. Fitting $\hat{\mu}_k$ and
$\hat{\Sigma}_k$ on chips that contain the plume absorbs part of the signal:
with the true $\mu$ and $\Sigma$ the same mask gives +1.7 % (the notebook
runs both and asserts both). That is a
retrieval choice, not a packaging one, and is out of scope here.

(r16-stage-6)=
### Stage 6 · The seams, as they stand (pipekit, xrtoolz, geotoolz-patcher)

**TL;DR.** Exercise the four places where today's packages disagree; each
becomes a workstream.

**Problem.** A `Signature` is a symbolic record of named dimensions and a
dtype, $\mathrm{sig} = ((d_1, n_1), \ldots, (d_r, n_r); \tau)$, that
`Sequential.summary` threads through every step. A mean over axis 0 maps
$((d_1, n_1), \ldots, (d_r, n_r)) \mapsto ((d_2, n_2), \ldots, (d_r, n_r))$.

```{code-block} python
:caption: The seams the workstreams fix, run against the pinned revisions.
:linenos:

# Shapes: T = 24, 5 × 5 grid
wind_mean = Sequential([WindSpeed(u="u10", v="v10", name="ws10"),
                        MeanScalar(axis=0)])                       # today
# (a) two Signature classes                                           → workstream A
wind_mean.summary(xrcore.Signature({"time": 24, "latitude": 5, "longitude": 5}))
# → ValueError: too many values to unpack (expected 2)               today
wind_mean.summary(pipekit.Signature(dims=(("time", 24), ("latitude", 5),
                                           ("longitude", 5))))
# → last step: Signature((latitude=5, longitude=5) float64)           today

# (b) the window dataset's stencil import                              → workstream D
import geopatcher.time
# → ModuleNotFoundError: No module named 'geopatcher.time'           today
from pipekit_train import XarrayWindowDataset
# → ImportError: "XarrayWindowDataset requires `geopatcher` … not published
#   on PyPI … `uv pip install -e ../geopatcher`"                      today

# (c) a pure-numpy stencil import loads the raster stack               → workstream D
import geopatcher.temporal.stencils   # sys.modules gains georeader, rasterio, shapely   today

# (d) base installs                                                    → workstreams C, F
importlib.metadata.requires("xrtoolz")   # includes xrtoolz-reader>=0.0.2       today
importlib.metadata.requires("geotoolz")  # includes pipekit>=0.0.2 (another project on PyPI)
```

The `XarrayWindowDataset` message is wrong as well as unlucky: geopatcher is
installed, and the module it wants was renamed. [Report 11](11_pipekit_train.md)
records the same break as its A5.

(r16-audit)=
## Current-state audit

Six overlaps need action; seven look like duplication and are not. The
audit read `references/pipekit` at `ef3c17f`, `references/geotoolz` at
`088475e` and `references/xrtoolz` at `308c7f2` on 14 October 2026.

:::{figure} ../images/reports/r16_readiness.svg
:label: fig-r16-readiness
:alt: A grid with one row per stage of the running example and per workstream, and one square per stack call in that row's sketches, coloured by its marker. Of 42 calls, 29 run today, 11 are proposed by this report's workstreams and 2 are code the reader writes.

What the page's sketches call, read from their markers: one square per stack
call. Of 42, 29 run today, 11 are proposed by this report's workstreams and
2 are code the reader writes. A row without squares has no marked call;
labels on the proposed squares name the workstream item.
:::

:::{table} Overlaps across the three repositories.
:label: tbl-r16-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| `Signature` | `pipekit/signature.py:29` (ordered pairs); `xrcore/signature.py:64` (mapping) | pipekit-array `reduce.py`; 18 constructions in 5 modules of xrtoolz and xrtoolz-einx | **Split ownership.** Two layouts; mixing them in one `Sequential` fails (stage 6a) |
| `nested_config`, `callable_name`, `require_operator` | `pipekit/_base/operator.py:282, 293, 302` | geotoolz `_src/config.py`, `augment/_src/operators.py`; geopatcher `integrations/pipekit.py` | **Split ownership.** Private API used by two packages |
| `JaxModelOp` ×2, `EquinoxModelOp`, `ModelOp` ×2, `SklearnModelOp`, `SklearnOp` ×2 | `pipekit_jax/model_op.py:42`; `pipekit_train/adapters/equinox.py:66`; `xrtoolz/inference/modelop.py:30, 227, 257`; `xrsklearn/_src/operator.py:32`; `geotoolz/learn/_src/operators.py:45, 469` | `EquinoxModelOp` at 13 sites in pipekit-train; `SklearnModelOp` at 4 | **True duplicate.** Three backends, eight classes, one name for two unrelated classes |
| `Variable`, `REGISTRY`, `apply_cf_attrs`, `subset_bbox`, `subset_time`, `subset_where` | `xrreader/types/_src/` (2,341 lines; NumPy, pandas, xarray and the standard library) | 8 imports in 7 xrtoolz modules (`atm`, `geo`, `viz`) | **Split ownership.** Vocabulary in the downloader; `xrtoolz` requires `xrtoolz-reader` |
| `XarrayWindowDataset`, `_xreader.training_iterator` | `pipekit_train/xarray_window.py:134`, `_xreader.py:347` | pipekit-train users | **Split ownership.** Imports `geopatcher.time` (`xarray_window.py:35`, `_xreader.py:380`), removed by geotoolz #431 |
| `open_emit_ch4_l2b`, `open_ghgsat_ch4_l2`, `open_tropomi_ch4_l2` · `goes`, `himawari`, `carbonmapper` readers | `xrreader/_src/local/openers/` (379 lines) · `geoproducts/` | xrreader users · geoproducts users | **Split ownership.** Two homes for mission readers, no rule between them (Report 3, question 7) |
| `exact_quotient` · `count_steps` | `geopatcher/_src/temporal/stencils.py:55` · `geocatalog.grid` | temporal stencils · `GeoSlice` alignment | **Healthy.** Named for what they do since #434; the draft's `divide_evenly` clash is gone |
| obstore pool, `files`, `cache`, `cog`, `hdf` | `geocloud` | geocatalog staging, geopatcher `CogField`, geotoolz `io.ReadHDF` / `ReadNetCDF`, geoproducts HDF5 | **Healthy.** One pool per process, one home since #431 |
| `pipekit.Cache` · `geopatcher.run.PatchCache` · `geocloud.cache` · xrreader `cache_path` | four packages | | **Healthy.** Operator results, patches across runs, local copies of remote objects, downloads: four jobs |
| `geoproducts.goes.TrueColor` · `geotoolz.viz.TrueColor` | `goes/presets.py:152` · `viz/_src/operators.py:124` | | **Healthy.** The preset binds ABI channels to the geotoolz operator and imports it lazily |
| `geotoolz.einx` · `xreinx` | both lanes | | **Healthy.** Each speaks its carrier: transform survival on GeoTensor, dimension names on xarray |
| geocatalog `Source` · xrreader `DataSource` | `geocatalog.sources` · `xrreader/_src/base.py` | | **Healthy.** File inventories with footprints against server-side subsets |
| `pipekit.Retry` · `geocatalog.utils.retry_transient_io` | | | **Healthy.** Retry any operator in a graph against retry one remote read |
:::

Smaller findings:

- **The workspace roots pin pipekit two ways, and they collide.** geotoolz's
  `[tool.uv.sources]` pins pipekit to commit `ef3c17f`; xrtoolz's follows
  `main` with no revision. Installing both editable into one environment
  fails with "conflicting URLs for package `pipekit`" unless sources are
  switched off (`uv pip install --no-sources`), which is how this page's
  environment was built.
- **pipekit-train's zarr floor is too low.** Its `cache`, `s3`, `gcs` and
  `xreader` extras allow `zarr>=2.18`, but `dataset.py:478` calls
  `zarr.storage.FsspecStore`, which exists only in zarr 3.
- **xrtoolz names an extra it does not have.** `ocn/_src/kinematics.py`
  says `pip install xrtoolz[oceanography]`; the extras are `wavelets`,
  `image`, `maps` and `vector`.
- **Only geotoolz runs a packaging test.** geotoolz, geotoolz-catalog and
  geotoolz-patcher have `tests/test_packaging.py`, and CI installs all five
  geotoolz packages with no extras; xrtoolz and pipekit have neither, which
  is how the last two findings shipped. geotoolz-cloud and geotoolz-products
  lack the test too.
- **xrtoolz's design docs predate pipekit and geopatcher.**
  `docs/design/boundaries.md` still assigns `Operator` to an in-repo `core`,
  tiling to `xrpatcher` and ensemble DA to `ekalmX`, and
  `ob-1.1-xrpatcher-integration.md` is still "proposed". `xrtoolz.rs` and
  `xrtoolz.ice` export nothing ([Report 3](03_sister_libraries.md#r3-todos)).
- **`import geopatcher.temporal.stencils` loads georeader, rasterio and
  shapely.** The package root imports `fields` and `RasterField`, and
  `_src/fields/raster.py:23`, `fields/reproject.py:20` and
  `_src/matched/field.py:50` import georeader and rasterio (which brings
  shapely) at module scope, though the stencils are pure NumPy.
- **None of the 18 distributions is on PyPI**, and every name is free except
  `pipekit` (version 0.3.5 there belongs to another author).

(r16-ws-a)=
## Workstream A · pipekit owns the composition contract

**Decision.** There is one `Signature`, pipekit's, and the helpers other
packages import from pipekit are public.

:::{table} Workstream A's moves.
:label: tbl-r16-move-a

| Symbol | Destination | Reason |
|---|---|---|
| `Signature` (ordered `(name, size)` pairs) | stays in `pipekit.signature` | positional operators (pipekit-array) need axis order |
| `Signature(dims)` accepting a mapping as well as pairs, `Signature.from_sizes(mapping, dtype)`, `.sizes` (read-only mapping) | `pipekit.signature`, new | xrtoolz builds and compares signatures by name; a `dict`'s order is the axis order, so `xrcore.Signature({"time": 24, …})` keeps its call semantics, not only its import |
| dtype canonicalisation without NumPy: a table of the accepted spellings, each mapped to the name `np.dtype(x).name` gives (`"f8"`, `"<f8"`, `"float64"`, `float` → `"float64"`; `"f4"`, `"<f4"` → `"float32"`; the integer, unsigned, bool and complex codes likewise, with any byte-order prefix); a dtype object gives its `.name` and a scalar type its `__name__`, both then looked up in the same table; anything else raises `TypeError` naming the input | `pipekit.signature` | xrcore's one behaviour pipekit lacks; pipekit-core stays dependency-free, and gate A1 checks the table against `np.dtype(x).name` for every entry, so equal dtypes give equal signatures |
| `xrcore.Signature` | re-export of `pipekit.Signature` | `from xrcore import Signature` keeps working |
| `nested_config`, `callable_name` | `pipekit.config` | used by geotoolz and geopatcher |
| `require_operator` | `pipekit` | used by geotoolz `augment` |
:::

1. **Widen pipekit's class.** Let the constructor take a mapping as well as
   ordered pairs (a mapping is converted to pairs in insertion order), and add
   `from_sizes`, `sizes` and dtype canonicalisation; `dims` stays ordered
   pairs. Existing `xrcore.Signature({...})` calls keep working unchanged.
2. **Re-export it from xrcore.** Replace `xrcore/signature.py` with the
   re-export, construct with `Signature.from_sizes(ds.sizes)`, and compare
   with `a.sizes == b.sizes` where order should not matter.
3. **Publish the helpers.** Add the three names to pipekit's public surface
   and replace the three private import statements.

```{code-block} python
:caption: Stage 6a, before and after workstream A.
:linenos:

# Shapes: T = 24, 5 × 5 grid
# Today:
sig = xrcore.Signature({"time": 24, "latitude": 5, "longitude": 5})   # today
wind_mean.summary(sig)                       # → ValueError               today
# After:
# {d: n} in insertion order ↦ ((d₁, n₁), …, (d_r, n_r))
sig = xrcore.Signature({"time": 24, "latitude": 5,
                        "longitude": 5})     # the same call     proposed (A)
xrcore.Signature is pipekit.Signature        # → True               proposed (A)
sig == pipekit.Signature.from_sizes(sig.sizes)   # → True           proposed (A)
wind_mean.summary(sig)                       # last: (latitude=5, longitude=5)  proposed (A)
```

This revisits nothing in [Report 2](02_pipekit_surface.md): its
`Signature` stays, and only gains the mapping view
[Report 3](03_sister_libraries.md) found missing.

(r16-ws-b)=
## Workstream B · One model-operator family

**Decision.** Each lane owns how its carrier becomes an array and back; each
backend is wrapped once.

:::{table} Workstream B's moves.
:label: tbl-r16-move-b

| Symbol | Destination | Reason |
|---|---|---|
| `pipekit_jax.JaxModelOp` | stays: the one Equinox wrapper | weight round-trip through the model registry |
| `pipekit_train.adapters.equinox.EquinoxModelOp` | deprecated alias of the above; the adapter returns `JaxModelOp` | [Report 11](11_pipekit_train.md#r11-ws-b)'s B1; it exists only so pipekit-train avoids pipekit-jax |
| pipekit-cycle in pipekit-jax's base | `pipekit-jax[diffrax]` | only `diffrax_model.py` needs it, for `RecursiveCheckpointAdjoint`; this removes the reason for the twin |
| `xrtoolz.inference.ModelOp` | stays: xarray marshalling for any callable | `method=None` means "call the model" |
| `xrtoolz.inference.JaxModelOp(model, *, jit=True, **kw)` | deprecated wrapper: warns, wraps `model` in `jax.jit`, imported lazily, when `jit` is true (today's default and today's implementation), and builds `ModelOp(model, method=None, **kw)` | answers [Report 3](03_sister_libraries.md#r3-open-questions)'s question 4; existing calls with `jit=` keep working for the deprecation period, and the wrapper adds no dependency: xrtoolz declares neither JAX nor Equinox, and Equinox stays out |
| `xrtoolz.inference.SklearnModelOp` | deprecated alias of `xrsklearn.SklearnOp` | one xarray scikit-learn bridge |
| `geotoolz.learn.ModelOp`, `SklearnOp` | stay | GeoTensor marshalling |
:::

1. **Return `JaxModelOp` from the Equinox adapter**, as Report 11's B1 says,
   and alias `EquinoxModelOp` to it.
2. **Move pipekit-cycle into pipekit-jax's `[diffrax]` extra**, so
   pipekit-train's `[equinox]` extra can depend on pipekit-jax.
3. **Alias the two xrtoolz wrappers** and teach `ModelOp` `method=None`.

```{code-block} python
:caption: A per-pixel Equinox emulator on the stage 4 cube, before and after.
:linenos:

# Shapes: X (T, 5, 5, f = 4) features; f_θ : ℝᶠ → ℝᵏ, k = 1
# Today:
op = xrtoolz.inference.JaxModelOp(net, feature_dim="feature")     # today
# After:
model = pipekit_jax.JaxModelOp(net)          # serialize_weights()  today
# stack non-feature dims: (T, 5, 5, f) → (n = T·25, f);  ŷₙ = f_θ(xₙ): (n, k)
op = xrtoolz.inference.ModelOp(model, method=None, feature_dim="feature",
                               output_dim="output")             # proposed (B)
# → (T, 5, 5, output = 1)
```

(r16-ws-c)=
## Workstream C · xrtoolz-core owns the shared xarray vocabulary

**Decision.** The names every xarray operator speaks live in xrtoolz-core;
xrtoolz-reader keeps its archives and readers.

This corrects one point from the 7 October draft: xrtoolz does not import
`BBox` or `TimeRange` from xrreader. It imports `Variable`, `REGISTRY`,
`apply_cf_attrs` and the three subset functions.

:::{table} Workstream C's moves.
:label: tbl-r16-move-c

| Symbol | Destination | Reason |
|---|---|---|
| `BBox`, `TimeRange`, `DepthRange`, `PressureLevels` | `xrcore.types` | request vocabulary, shared with every reader |
| `Variable`, `REGISTRY`, `resolve`, `register` | `xrcore.types` | CF variables used by `xrtoolz.atm`, `geo` and `viz` |
| `apply_cf_attrs` | `xrcore.types` | used by `xrtoolz.atm` |
| `validate_dataset`, `validate_variable`, `ValidationReport` | `xrcore.types` | the validators belong beside the registry they check against |
| `subset_bbox`, `subset_time`, `subset_where` | `xrcore.types` | used by `xrtoolz.geo` |
| `Request`, `Station`, `StationCollection` | stay in `xrreader.types` | reader-only |
| `xrtoolz-reader` in xrtoolz's base dependencies | a `[reader]` extra | the base install stops pulling the archive code |
| `xrtoolz.rs`, `xrtoolz.ice` | removed | empty; radiometry and indices are geotoolz's (rule 12) |
:::

1. **Move the modules** from `xrreader/types/_src/` into
   `xrcore/types/`. They need NumPy, pandas, xarray and the standard
   library. pandas is in xrtoolz-core's tree only through xarray today, so
   the move adds `pandas` to xrtoolz-core's `dependencies`: a package
   declares what it imports (rule 8).
2. **Re-export from `xrreader.types`** for the 0.x series.
3. **Repoint xrtoolz's eight imports**, move xrtoolz-reader to `[reader]`,
   and delete the empty `rs` and `ice` packages.

```{code-block} python
:caption: Stage 4's imports, before and after workstream C.
:linenos:

# Today:
from xrreader.types import Variable, apply_cf_attrs, subset_bbox      # today
# After:
from xrcore.types import Variable, apply_cf_attrs, subset_bbox        # proposed (C)
from xrreader.types import Variable          # still works, same object      proposed (C)
```

(r16-ws-d)=
## Workstream D · geotoolz-patcher owns windows for every carrier

**Decision.** Windowed sampling for training lives with the stencils that
define it, in geotoolz-patcher, and nothing in pipekit imports a geo package.

:::{table} Workstream D's moves.
:label: tbl-r16-move-d

| Symbol | Destination | Reason |
|---|---|---|
| `from geopatcher.time import …` | `from geopatcher.temporal.stencils import …` | D1: the module was renamed in #431 |
| `XarrayWindowDataset`, `_xreader` | `geopatcher.integrations.train`, behind `geotoolz-patcher[train]` (pipekit-train, xarray, zarr, Grain) | D2: removes the only pipekit → geo edge |
| `from pipekit_train import XarrayWindowDataset` | removed, with no shim; the notebooks of Reports 11 and 15, which import it through a `geopatcher.time` alias, already try `geopatcher.integrations.train` first | it raises `ImportError` without that alias (stage 6), and a re-export would bring back the pipekit → geo edge D2 removes |
| the raster fields, reprojection, matched fields and the `RasterDomain` alias | imported on first use from the package root | D3: a stencil import should load NumPy only |
| `xrpatcher` in xrtoolz's design docs | geopatcher's `XarrayField`, `GridSampler`, `MergePatches` | D4: one patcher for both lanes |
:::

### D1 · Fix the import

This is [Report 11](11_pipekit_train.md#r11-ws-a)'s A5: import from
`geopatcher.temporal.stencils` and make the error name the missing module.
It ships first, because the dataset does not import at all today.

### D2 · Move the dataset

`XarrayWindowDataset` subclasses pipekit-train's `TrainingDataset` and rides
its Grain path, so it needs pipekit-train; the stencils need geopatcher. The
package that knows both is the one with the integration modules already:
geopatcher has `integrations/pipekit.py` for operator graphs, and gains
`integrations/train.py`. Two callers in this repository use the old path: the
notebooks of Reports 11 and 15. Both import from `geopatcher.integrations.train`
when it exists and fall back to pipekit-train otherwise, so they keep running
on either side of D2.

### D3 · A cheap stencil import

The package root imports the raster field, reprojection and matched fields,
which import georeader, rasterio and shapely at module scope. Load them on first use
from the root, as geocatalog already does for its extras, so
`geopatcher.temporal.stencils` loads NumPy and nothing heavier. This is the
lazy half of [Report 7](07_geopatcher.md)'s question 3; moving them behind a
`[raster]` extra is the other half, and stays open.

### D4 · The xarray lane uses geopatcher

Mark `ob-1.1-xrpatcher-integration.md` superseded, rewrite
`boundaries.md`'s ownership map from this page's rules, and change the
`interpolate/_src/downscale.py` docstring to compose with `GridSampler` and
`MergePatches`.

```{code-block} python
:caption: Training windows from the stage 4 cube, before and after workstream D.
:linenos:

# Shapes: T = 24 hourly steps; input window 6 h, target window 1 h
# Today:
from pipekit_train import XarrayWindowDataset      # → ImportError        today
# After:
from geopatcher.temporal.stencils import TimeStencil                  # today
from geopatcher.integrations.train import XarrayWindowDataset   # proposed (D2)
# x = (u, v)[t₀−5h … t₀]: (2, 6, 5, 5);  y = (u, v)[t₀+1h]: (2, 1, 5, 5)
ds = XarrayWindowDataset(
    source={"input": era5, "target": era5},
    stencils={"input": TimeStencil(start="-5h", stop="0h", step="1h", closed="both"),
              "target": TimeStencil(start="1h", stop="1h", step="0h", closed="both")},
    store_version="stage4-wind", split_guard=0,
)                                                               # proposed (D2)
x, y = ds[0]                         # (2, 6, 5, 5), (2, 1, 5, 5)  proposed (D2)
# valid origins: |𝒪| = T − a₋ − a₊ = 24 − 5 − 1 = 18         today (stencils)
```

The origin count follows from the stencils: with $a_- = 5$ steps of history
and $a_+ = 1$ step of lead, an origin $t_0$ is valid only when both windows
fit inside the $T = 24$ steps, which today's `valid_origin_points` confirms
on the stage 4 cube:

$$
|\mathcal{O}| = T - a_- - a_+ = 18 .
$$ (r16-eq-origins)

(r16-ws-e)=
## Workstream E · A mission reader lives with its output carrier

**Decision.** A reader whose output is a georeferenced raster (an affine
transform and a CRS) belongs to geotoolz-products; one whose output is a
swath, 2-D latitude and longitude, profiles or stations belongs to
xrtoolz-reader.

This answers [Report 3](03_sister_libraries.md#r3-open-questions)'s
question 7: the TROPOMI reader stays on the xarray side.

:::{table} Workstream E's placements.
:label: tbl-r16-move-e

| Reader | Output | Lives in |
|---|---|---|
| GOES-R ABI, Himawari AHI, Carbon Mapper, toy sensor | gridded `GeoTensor` | geotoolz-products (today) |
| TROPOMI CH₄ L2 | `(scanline, ground_pixel)` swath | xrtoolz-reader (today) |
| GHGSat CH₄ L2 | `(y, x)` with 2-D latitude and longitude | xrtoolz-reader (today) |
| EMIT L2B CH₄ enhancement, NetCDF container | `(time, y, x)` with 2-D latitude and longitude | xrtoolz-reader (today) |
| EMIT L2B CH₄ enhancement, distributed GeoTIFF | gridded `GeoTensor` | geotoolz-products, new: xrreader's opener declines this format |
:::

1. **Write the rule** into both packages' contributing guides and
   `AGENTS.md`.
2. **Add the EMIT L2B GeoTIFF reader** to geotoolz-products, the one format
   the rule assigns there that neither package reads today.
3. **Index both packages' subpackages** in the capability index, so a
   mission name read in two places is visible
   ([Report 3](03_sister_libraries.md#r3-todos)).

```{code-block} python
:caption: One EMIT plume through each reader, after workstream E.
:linenos:

# Shapes: H × W ortho grid
enh = geoproducts.emit.Reader(tif_path).read()        # GeoTensor (H, W)   proposed (E)
ds = xrreader.open_emit_ch4_l2b(nc_path)               # Dataset (time, y, x)  today
```

(r16-ws-f)=
## Workstream F · Packaging and release

**Decision.** The stack publishes under names it owns, pins itself by version
range, and every repository tests its own packaging.

### F1 · Names

pipekit publishes as `pipekit-core` and keeps `import pipekit`. This
revisits [Report 2](02_pipekit_surface.md)'s settled question 1: the import
name stays `pipekit`, but the distribution cannot, because `pipekit` on PyPI
belongs to another project. The same decision serves the GeoML stack, which
also depends on pipekit. Answering [Report 6](06_geocatalog.md) and
[Report 7](07_geopatcher.md): the public names are `geotoolz-catalog` and
`geotoolz-patcher`, as the monorepo already builds them.

Recommended: `pipekit-core`. Alternative: a new import name as well, which
renames every import in three repositories for no functional gain.

### F2 · Pins and floors

1. **Replace git pins with version ranges** once a tier is on PyPI, and drop
   both workspace roots' `pipekit` entries from `[tool.uv.sources]`.
2. **Raise pipekit-train's zarr floor** to `zarr>=3.0` in its `cache`, `s3`,
   `gcs` and `xreader` extras.
3. **Fix the xrtoolz hint**: add `oceanography = ["gsw"]` or change it to
   `pip install gsw`.

### F3 · Packaging tests

Copy geotoolz's `tests/test_packaging.py`, which fails when a module-level
import is missing from `[project.dependencies]`, and its base-install CI job
into every xrtoolz and pipekit package, and the test into geotoolz-cloud and
geotoolz-products. That test catches undeclared module-level imports only, so
two more checks cover the faults F2 fixes:

1. **A minimum-versions job** installs each extra at its floors
   (`uv pip install --resolution lowest-direct`) and runs a test that calls
   the code path the extra exists for. For pipekit-train's `cache` extra that
   is `CachedDataset` with an fsspec URL as its cache directory
   (`memory://…`, so no cloud is needed), which opens the store through
   `zarr.storage.FsspecStore` (`dataset.py:478`) and fails on today's
   `zarr>=2.18` floor; a local path never reaches that line.
2. **An install-hint test** collects every `pip install <dist>[<extra>]`
   string in a package's source and checks that `<extra>` is declared in its
   `[project.optional-dependencies]`, which fails on today's
   `xrtoolz[oceanography]` hint.

### F4 · Release in tier order

:::{table} Release order, by base dependencies. An extra may name a later tier; it resolves once that tier is published, and gate F4 checks the whole set.
:label: tbl-r16-release

| Release | Distributions | Base dependencies in the stack |
|---|---|---|
| 1 | `pipekit-core`, `geotoolz-cloud`, `geotoolz-catalog`, `geotoolz-patcher`, `geotoolz-products`, `xrtoolz-grad` | none |
| 2 | `pipekit-array`, `-cycle`, `-evaluate`, `-experiment`, `-jax`, `-train`, `xrtoolz-core`, `geotoolz` | `pipekit-core` (pipekit-jax after B) |
| 3 | `xrtoolz-einx`, `xrtoolz-sklearn`, `xrtoolz-reader` | `xrtoolz-core` (xrtoolz-reader after C) |
| 4 | `xrtoolz` | the xrtoolz family |
:::

### F5 · One stack check

A scheduled job installs all 18 distributions from PyPI into one fresh
environment, with `geotoolz-patcher[train]` (which brings pipekit-train's
Grain and zarr extras), and runs each package's fast tests and this page's
notebook with `REPORT16_STATE=target`. In that mode the notebook checks the seams of
stage 6 against the realigned stack (one `Signature`, the window dataset
importing from geopatcher and building 18 windows of the right shapes from
stage 4's cube, a NumPy-only stencil import, no reader in
xrtoolz's base install, and none of the 18 distributions still requiring
`pipekit` by name, which after F1 is another project that pip would install
beside `pipekit-core` without a conflict); by default it checks them as they stand today, so
the page's stage 6 stays reproducible at the pinned revisions.

(r16-gates)=
## Testing gates

Each workstream merges only when its gates pass; the K gates are checks the
code passes today and must keep passing.

:::{table} Gates for the proposals and for what ships.
:label: tbl-r16-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 One Signature | A | `xrcore.Signature is pipekit.Signature`, and stage 6a's `summary` with a signature built by today's call, `xrcore.Signature({"time": 24, "latitude": 5, "longitude": 5})`, ends in `(latitude=5, longitude=5)` (today `ValueError`); and for every spelling in the dtype table (`"f8"`, `"<f4"`, `float`, `"i4"`, `"?"`, …), `Signature.from_sizes(sizes, dtype=x).dtype == np.dtype(x).name`, with NumPy used only by the test |
| A2 Signature round-trip | A | every existing pipekit, pipekit-array and xrtoolz signature test passes against the merged class, with 0 failures |
| A3 No private imports | A | `grep -r "pipekit\._" ` over geotoolz returns 0 lines (today 3) |
| B1 One Equinox wrapper | B | pipekit-train's tests pass with the adapter returning `JaxModelOp`, and `pip install pipekit-jax` without extras installs no pipekit-cycle |
| B2 Alias parity | B | each deprecated alias returns outputs equal to its replacement to 0 on xrtoolz's inference tests and emits exactly one `DeprecationWarning`; `JaxModelOp(net, feature_dim="feature", jit=True)` and `jit=False` both construct and agree to 0, in an environment with JAX and without Equinox |
| C1 Reader-free xrtoolz | C | `importlib.metadata.requires("xrtoolz")` lists no `xrtoolz-reader` outside `[reader]` (today it does), and `import xrtoolz` succeeds without it |
| C2 Vocabulary shims | C | for every name the move table sends to `xrcore.types` (`BBox`, `TimeRange`, `DepthRange`, `PressureLevels`, `Variable`, `REGISTRY`, `resolve`, `register`, `apply_cf_attrs`, `validate_dataset`, `validate_variable`, `ValidationReport`, `subset_bbox`, `subset_time`, `subset_where`), `getattr(xrreader.types, n) is getattr(xrcore.types, n)`, so old imports and `isinstance` checks keep working |
| D1 Window import | D | `XarrayWindowDataset` imports with the pinned geopatcher and no shim (today `ImportError`); Report 11's stage 8 pairs satisfy $y = M x_{-1}$ to 1e-5 |
| D2 No upward edges | D | grepping every pipekit package for `geopatcher`, `geocatalog`, `geotoolz` or `xrcore` returns 0 lines (today 2 files); the dataset's `content_hash` on stage 4's cube is identical before and after the move |
| D3 Cheap stencil import | D | in a fresh interpreter, `import geopatcher.temporal.stencils` puts none of `georeader`, `rasterio`, `shapely`, `pyproj`, `geopandas`, `xarray`, `pandas`, `scipy`, `dask` or `zarr` in `sys.modules` (today the first three) |
| E1 Carrier rule | E | the capability index lists no mission name under both geoproducts and xrreader except EMIT L2B, whose GeoTIFF and NetCDF containers are different carriers by design (stage table above); and those two EMIT readers agree on a fixture's enhancement to 1e-6 |
| F1 Names | F | `pipekit-core`, `geotoolz-cloud`, `geotoolz-products`, `geotoolz-catalog` and `geotoolz-patcher` resolve to this stack on PyPI |
| F2 One environment | F | geotoolz and xrtoolz install editable into one environment without `--no-sources` (today a URL conflict) |
| F3 Packaging tests | F | xrtoolz and pipekit run `test_packaging.py` and a base-install job; the minimum-versions job fails on today's `zarr>=2.18` floor at `FsspecStore` and the install-hint test fails on today's `[oceanography]` hint, and both pass after F2 |
| F4 Clean install | F | all 18 distributions install from PyPI into one fresh environment with no git URLs and no resolver conflict |
| F5 Stack check | F | the scheduled job passes on the latest release of every distribution, with the notebook's `REPORT16_STATE=target` seam checks |
| K1 Noise floor (existing) | — | stage 2: the background standard deviation of $\hat{\alpha}$ is within 5 % of $\sigma_\alpha = (t^\top \Sigma^{-1} t)^{-1/2}$. Passes today: 107.2 against 103.9, +3.2 % |
| K2 Split and merge (existing) | — | stage 2: the identity operator through `GridSampler`, `ApplyToChips` and `MergePatches` returns the scene to 1e-5. Passes today: 0.0 |
| K3 Column mass (existing) | — | stage 3: `ColumnToMass` equals {eq}`r16-eq-mass` to 1e-12. Passes today |
| K4 Emission rate (existing) | — | stage 5: `IMEEstimate`'s rate equals $U_{\mathrm{eff}}\,\mathrm{IME}/\ell$ to 1e-12. Passes today |
:::

(r16-risks)=
## Sequencing and risks

The name and the contract go first: every later change either ships under the
new name or touches operators that carry signatures. B, C, D and E are
independent of each other and run side by side; publishing waits for all of
them.

:::{figure} ../images/reports/r16_roadmap.svg
:label: fig-r16-roadmap
:alt: Phase 1: F1 publish as pipekit-core and A one Signature with public helpers, then gates F1, A1, A2, A3. Phase 2: B model operators, C vocabulary into xrtoolz-core, D datasets into the patcher, E the carrier rule for mission readers, side by side, then gates B1, B2, C1, C2, D1, D2, D3, E1. Phase 3: F2 to F5, pins and floors, packaging tests, publishing in tier order and the stack check, then their gates.

The name and the contract first, then four independent moves, then the
release.
:::

- **Ordered signatures break order-free comparisons.** xrcore's equality
  ignores dimension order; pipekit's does not. Tests that build signatures
  from differently ordered dicts start failing. Fallback: rewrite those
  comparisons to `a.sizes == b.sizes` in the same PR; gate A2 finds any left.
- **Moved names break downstream code.** Every move in B, C and D leaves a
  deprecated alias or re-export for the 0.x series, except
  `pipekit_train.XarrayWindowDataset`, which cannot be imported today and
  whose re-export would undo D2. Fallback: if an alias
  cannot be kept (a class that changes base), pin the old minor version in
  the dependent package until it migrates.
- **The other `pipekit` shares the import name.** If its wheel also installs
  a top-level `pipekit` module, the two cannot share an environment.
  Fallback: say so in the README; renaming the import is not proposed.
- **Caches keyed on the dataset could go stale.** A move that changed the
  content hash would orphan every cache built on it. It does not:
  `content_hash` hashes the literal string `"XarrayWindowDataset"` and the
  dataset's own config, not its module, and gate D2 checks the hash is
  unchanged. Serialised pipelines are not at risk, because the dataset sets
  `forbid_in_yaml` and never round-trips through `pipekit.serial`.
- **Lazy imports hide a missing dependency.** D3 makes the raster domain
  load on first use, so an install without georeader fails later than
  today. Fallback: geotoolz-patcher's packaging test already installs with no
  extras; extend it to build each field once.
- **The carrier rule meets a hybrid product.** A product shipped both as an
  orthorectified raster and as a swath (EMIT) gets a reader in each package.
  Fallback: that is the rule working; the capability index shows both, and
  gate E1 checks they agree.

(r16-open-questions)=
## Open questions

1. **Does geoproducts get an xarray output?** A `to_xarray` on its readers
   would let the xarray lane use gridded missions without a second reader,
   at the cost of xarray in geotoolz-products' base.
2. **Should geotoolz-cloud's base drop georeader?** It needs georeader only
   for the `GeoTensor` that `cog` and `hdf` return; an index-only or
   bytes-only install would move those modules behind an extra, as
   [Report 7](07_geopatcher.md) asks for the patcher.
3. **Does xrtoolz-reader use geotoolz-cloud for its downloads?** Its cache
   and the cloud package's `cache` keep local copies by different keys
   (request against URI). Sharing one would cross the lane rule at tier 2,
   which the tiers allow.

(r16-todos)=
## Upcoming work

:::{table} Follow-ups outside the workstreams.
:label: tbl-r16-todos

| Follow-up | Where | Why |
|---|---|---|
| Point the 7 October draft at this page | the "Geo data stack" Claude doc | this page is the source of truth |
| Rewrite `boundaries.md` from this page's ownership rules | xrtoolz `docs/design/` | it predates pipekit and geopatcher (D4) |
| Fix the misleading `XarrayWindowDataset` error message in the same PR as D1 | pipekit-train `xarray_window.py:40` | geopatcher is installed; the module was renamed |
| Index geoproducts, geocloud and the xrtoolz subpackages | geostack `scripts/capabilities.py` | E1 and the `JaxModelOp` clash need them ([Report 3](03_sister_libraries.md#r3-todos)) |
:::

(r16-references)=
## References

:::{note} Companion reports
:class: simple
Reports [2](02_pipekit_surface.md), [3](03_sister_libraries.md),
[6](06_geocatalog.md), [7](07_geopatcher.md) and [11](11_pipekit_train.md)
are published.
:::

### Citations

The matched filter follows Thompson et al. [@thompson2015], its covariance
the shrinkage estimator of Ledoit and Wolf [@ledoit2004], and the effective
wind the calibration of Varon et al. [@varon2018].

### Sources

- pipekit, xrtoolz and geotoolz at the revisions pinned in `references/`:
  their `pyproject.toml` files, public `__init__` modules, call sites,
  design docs and changelogs (geotoolz #420, #431 and #434 for the changes
  since the draft).
- The running example's script and outputs, described in [](#r16-running).
- The 7 October draft of this plan, which this page replaces.
