---
title: The Beer–Lambert column retrieval
subtitle: radtrax · 1 · From top-of-atmosphere reflectance to a methane column enhancement, term by term
short_title: radtrax · 1 · Column retrieval
description: The single-layer Beer–Lambert model behind satellite methane retrievals, with where each of its terms comes from, the multispectral proxy and hyperspectral continuum baselines, and the look-up table that inverts it. The source note is checked against the stack's readers and operators and against a synthetic Sentinel-2 scene and EMIT-like cube.
keywords:
  - radtrax
  - radiative transfer
  - methane
  - Beer–Lambert
  - Sentinel-2
  - MBSP
  - matched filter
  - look-up table
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

A methane plume over a bright surface removes a little light in the
short-wave infrared, and the single-layer Beer–Lambert model turns that
deficit into a column mass. This page writes the model out term by term,
says where each term comes from in a satellite product, and follows the two
ways to estimate the plume-free baseline: a proxy band for multispectral
sensors such as Sentinel-2, and a continuum fit for hyperspectral ones such
as EMIT. It then inverts the model with a look-up table. The source note's
physics holds. Five of its details do not: Sentinel-2's band 11 is not blind
to methane, the effective cross-section is valid only for weak absorption,
Sentinel-2 L1C has a radiometric offset, Landsat L1 is not NetCDF or HDF5,
and the operational hyperspectral retrievals are matched filters, not
continuum fits. On a synthetic Sentinel-2 scene, the note's recipe recovers
80 % of the plume's mass; the multi-band ratio with a look-up table recovers
96 %. **radtrax**, the GeoModels package for radiative transfer in JAX
(formerly rtmx), would own the band transmittance, the table and its
inversion. Much of it exists already in plumax (band-integrated radiative
transfer, a HITRAN table pipeline, a matched filter), which the
[package boundaries](../package_boundaries.md) page proposes to move.

+++

:::{note} About this page
:class: dropdown

| | |
|---|---|
| **Status** | Design. radtrax does not exist yet; this page fixes the physics its first module implements, and names what the stack and plumax already do |
| **Reading time** | about 25 minutes |
| **Audience** | anyone writing or reviewing a methane column retrieval, and anyone deciding what radtrax owns |
| **Companions** | [Report 16](../../reports/16_geo_data_stack.md) (one methane scene through geotoolz, with the matched filter and the IME); [Report 5](../../reports/05_pipekit_jax.md) (pipelines under JAX, for a differentiable radtrax); [Report 17](../../reports/17_cycle.md) (the five-band sensor, a simpler Beer–Lambert model) |
| **Code checked against** | georeader-spaceml 2.4.1 (the Sentinel-2 L1C reader, its SRF table COPE-GSEG-EOPG-TN-15-0007 v4.0, the EMIT reader and the Gaussian SRF helper); geotoolz 0.10.2 at `088475e` (`plume`, `spectral`, `matched_filter`); xrtoolz 0.0.5 at `308c7f2` (`atm.gas.ch4`). Read on 14 October 2026 |
| **Running example** | every number and figure is from one run of `notebooks/geomodels/radtrax_01_column_retrieval.py`, which builds the scenes from fixed seeds and asserts each number the page shows |
| **Sources** | the source note, "The Core Forward Model Equation" and its four companion sections [@geostack_radtrax_note]; Varon et al. on Sentinel-2 [@varon2021]; Thompson et al. on the matched filter [@thompson2015] |
:::

:::{important} The spectrum is synthetic
HITRAN is not reachable from the environment that ran the example, so the
methane lines are drawn at random in the 1.66 μm and 2.3 μm bands and
scaled so that the band-averaged cross-sections are σ̄₁₂ = 0.40 m²/kg and
σ̄₁₁ = 0.04 m²/kg, a ratio of one tenth chosen for the example. The
Sentinel-2 response functions are the real ones. Every bias below has the
sign and the cause it would have on a real spectrum; its size is the size it
has on this one. Rerunning the notebook on HITRAN lines is the first item of
[upcoming work](#rx1-todos).
:::

:::{important} What changed from the source note
The source note's **forward model and its inversion stand**: Beer–Lambert
attenuation along a two-way path, a baseline from scene statistics, and a
table from a radiative transfer model. Five details **do not**. Band 11 is
**not blind to methane**: its response reaches 1668 nm, across the 1.66 μm
band. The **effective cross-section** is the weak-absorption limit of the
band transmittance, so the table is needed for saturation, not for "mild"
line-shape effects. **Sentinel-2 L1C** carries a radiometric offset since
2022. **Landsat L1** is GeoTIFF with scaling coefficients, not NetCDF or
HDF5. And the **hyperspectral** products that map methane operationally use
a matched filter, not a continuum fit. A **Differs** box marks each place.
:::

(rx1-model)=
## The forward model

The reflectance a sensor measures in band $b$ at pixel $(x, y)$ is the
surface reflectance times the transmittance of the two-way path through the
atmosphere, averaged over the band's spectral response $f_b(\lambda)$:

$$
R_b(x, y) = \rho_b(x, y) \,
\frac{\int f_b(\lambda) \, e^{-m(x, y)\, \sigma(\lambda)\, [\Omega_{\text{bg}} + \Delta\Omega(x, y)]} \, d\lambda}
     {\int f_b(\lambda) \, d\lambda} .
$$ (rx1-eq-exact)

Here $\sigma(\lambda)$ is the methane cross-section per unit mass,
$\Omega_{\text{bg}}$ the background column and $\Delta\Omega$ the plume's
enhancement. When every line is optically thin, the band average of the
exponential is the exponential of the band-averaged cross-section, and
{eq}`rx1-eq-exact` becomes the source note's model:

$$
R_b = \rho_b \, \tau_{\text{bg},b} \, e^{-m \, \bar\sigma_b \, \Delta\Omega} ,
\qquad
\bar\sigma_b = \frac{\int f_b \, \sigma \, d\lambda}{\int f_b \, d\lambda} ,
$$ (rx1-eq-weak)

so that the plume's optical depth is linear in the column:

$$
\tau_b = -\ln \frac{R_b}{\rho_b \, \tau_{\text{bg},b}} = m \, \bar\sigma_b \, \Delta\Omega .
$$ (rx1-eq-tau)

{eq}`rx1-eq-weak` is the right picture for small plumes and the wrong one
for strong lines. The running example measures how wrong in [stage 2](#rx1-stage-2).

(rx1-variables)=
## The variables

:::{table} Every term of the model, where it comes from, and the stack piece that supplies it today.
:label: tbl-rx1-vars

| Symbol | Name | Role | Units | From | In the stack today |
|---|---|---|---|---|---|
| $R_b(x, y)$ | top-of-atmosphere reflectance | input | dimensionless | the L1 product | georeader's Sentinel-2 L1C and EMIT readers |
| $\rho_b(x, y)$ | surface reflectance | nuisance | dimensionless | scene statistics, as part of the baseline | |
| $\tau_{\text{bg},b}(x, y)$ | background transmittance (methane, water vapour, CO₂) | nuisance | dimensionless | scene statistics, or a radiative transfer model | |
| $m(x, y)$ | two-way geometric air-mass factor | input | dimensionless | sun and view zenith angles in the metadata | georeader's `sza` and `vza` grids (Sentinel-2) |
| $\sigma(\lambda)$, $\bar\sigma_b$ | mass absorption cross-section, and its band average | parameter | m²/kg | a line list, then the band's response | georeader's `read_srf` (Sentinel-2) and `srf` (Gaussian); plumax's `hapi_lut` (HITRAN through HAPI) and `radtran.srf` |
| $\Omega_{\text{bg}}$ | background methane column | parameter | kg/m² | climatology or a model | xrtoolz's `dry_air_column` and `mixing_ratio_to_column` |
| $\Delta\Omega(x, y)$ | column mass enhancement | output | kg/m² | the inversion | geotoolz's matched filter; xrtoolz's `column_mass_to_delta_vmr` converts it |
:::

:::{warning} Differs from the plan
The source note gives $\sigma$ in m²/kg, which is what {eq}`rx1-eq-tau` needs
for a column in kg/m². HITRAN's cross-sections are per molecule, in
cm²/molecule; the conversion is $\sigma_{\text{mass}} = \sigma_{\text{mol}}
\, N_A / M_{\text{CH}_4} \times 10^{-4}$ m²/kg. The note also lists
$\tau_{\text{bg}}$ as a "parameter layer"; it is a nuisance the baseline
absorbs, like $\rho$, and never estimated on its own.
:::

(rx1-provenance)=
## Where each term comes from

:::{figure} ../../images/radtrax_01_column_retrieval/provenance.svg
:label: fig-rx1-provenance
:alt: Six rows. The satellite L1 product gives digital numbers or radiance, scaled to reflectance R_b. The metadata gives the sun and view zenith angles, combined into the air-mass factor m. The agency SRF table and a line list give the band cross-sections and transmittances. Scene statistics give the baseline from clean pixels or spectral windows. A radiative transfer model gives the look-up table.

The sources of the model's terms, from the notebook.
:::

**Reflectance.** For Sentinel-2 L1C the granule's JPEG 2000 bands are
top-of-atmosphere reflectance scaled to integers:

$$
R_b = \frac{\text{DN}_b + \text{RADIO\_ADD\_OFFSET}_b}{\text{QUANTIFICATION\_VALUE}} ,
$$ (rx1-eq-dn)

with a quantification value of 10,000 and, since processing baseline 04.00
(January 2022), an offset of −1,000. Hyperspectral L1 products are radiance
$L_b$, converted with the band's solar irradiance $E_b$ and the Earth–Sun
distance $d$ in astronomical units:

$$
R_b = \frac{\pi \, L_b \, d^2}{E_b \cos\theta_s} .
$$ (rx1-eq-rad)

**Air-mass factor.** The geometric two-way path through a plane-parallel
layer is

$$
m = \frac{1}{\cos\theta_s} + \frac{1}{\cos\theta_v} ,
$$ (rx1-eq-amf)

with $\theta_s$ and $\theta_v$ the sun and view zenith angles.

**Cross-section.** A line list gives $\sigma(\lambda)$ at the layer's
pressure and temperature; the band's response function averages it,
{eq}`rx1-eq-weak`, or, for the exact model, averages the transmittance,
{eq}`rx1-eq-exact`.

```{code-block} python
:caption: Reading the terms with the stack today.
:linenos:

# Shapes: S2 L1C tile (13, H, W); EMIT L1B (285, H, W); SRF table (877, 13)
s2 = S2ImageL1C(safe_path)                                              # today
s2.quantification_value(), s2.radio_add_offsets()  # 10000, {B12: -1000}  today
# R_b = (DN_b + offset_b) / Q
R = s2.load()                                     # reflectance           today
# m = 1/cos θ_s + 1/cos θ_v, from the 5 km angle grids
sza, vza = s2.sza, s2.vza["B12"]                  # GeoTensors            today
m = 1 / np.cos(np.radians(sza)) + 1 / np.cos(np.radians(vza))           # user
# R_b = π L_b d² / (E_b cos θ_s)
emit = EMITImage(l1b_path); R_h = emit.load(as_reflectance=True)        # today
srf = read_srf("S2A")              # B11, B12 at 1 nm, offline            today
```

:::{warning} Differs from the plan
Three corrections to the note's sources. **Sentinel-2 L1C** is
$(\text{DN} + \text{offset}) / 10{,}000$ for products from processing baseline
04.00 onwards, not $\text{DN} / 10{,}000$; georeader reads the offset from the
product metadata (`radio_add_offsets`). Its angles are 5 km grids in
`MTD_TL.xml`, interpolated to the pixels. **Landsat L1** is GeoTIFF, with
`REFLECTANCE_MULT_BAND_x` and `REFLECTANCE_ADD_BAND_x` in `MTL.txt` and a
division by the sine of the sun elevation; its `MTL.txt` holds only the
scene-centre sun angles, and per-pixel angles come from the angle bands.
**EMIT and PRISMA** deliver radiance, so {eq}`rx1-eq-rad` comes first;
EMIT's geometry is in a separate observation file, which georeader reads.
:::

(rx1-running)=
## The running example: one plume, two sensors

A 4 km × 4 km Sentinel-2 scene at 20 m, 200 × 200 pixels. The surface has
a smooth band 11 reflectance between 0.20 and 0.40 and two cover types, with
band 12 at 0.84 and 0.80 of band 11 and a 1 % per-pixel scatter. A Gaussian
plume, elongated and rotated by 30°, peaks at $\Delta\Omega = 0.3$ kg/m².
The background is 1.9 ppm, a column of 0.0109 kg/m², and the sun and view
zenith angles are 35° and 5°, so $m = 2.225$. Each band adds noise with a
standard deviation of 0.0015 in reflectance. [Stage 5](#rx1-stage-5) adds
an EMIT-like cube.

(rx1-stage-1)=
### Stage 1 · The forward model (georeader → radtrax)

**TL;DR.** Build the exact model {eq}`rx1-eq-exact` with Sentinel-2's own
response functions.

```{code-block} python
:caption: The forward model with Sentinel-2A's band 11 and band 12, from a real run.
:linenos:

# Shapes: G = 100,000 wavelengths (0.01 nm); H = W = 200
srf = read_srf("S2A")                     # (877, 13) at 1 nm         today
f11, f12 = np.interp(WL, srf.index, srf["B11"]), ...        # (G,)     user
# σ̄_b = ∫ f_b σ dλ / ∫ f_b dλ
s11, s12 = band_mean(sigma, f11), band_mean(sigma, f12)     # 0.04, 0.40  user
# T_b(m, Ω) = ∫ f_b exp(−m σ Ω) dλ / ∫ f_b dλ
T = radtrax.band_transmittance(sigma, f12)          # (m, Ω) → T   proposed (radtrax)
# R_b = ρ_b T_b(m, Ω_bg + ΔΩ) + ε_b
R12 = rho12 * T(m, OMEGA_BG + d_omega) + eps        # (H, W)       user
# T₁₂(m, Ω_bg) = 0.990: the background takes 1 % of band 12
```

Sentinel-2A's band 12 is above half its peak from 2113 to 2286 nm, and
band 11 reaches 1668 nm. The background methane already removes 1 % of
band 12's light along this path.

(rx1-stage-2)=
### Stage 2 · The effective cross-section against the exact band (radtrax)

**TL;DR.** Compare the linear optical depth {eq}`rx1-eq-tau` with the exact
one, the curve of growth.

**Problem.** The plume's exact band optical depth, above the background, is

$$
\tau_b(\Delta\Omega) = -\ln \frac{T_b(m, \Omega_{\text{bg}} + \Delta\Omega)}{T_b(m, \Omega_{\text{bg}})} ,
$$ (rx1-eq-cog)

and the linear model claims $\tau_b = m \bar\sigma_b \Delta\Omega$. A line's
core saturates long before its wings do, so the band absorbs less than the
linear model says.

:::{figure} ../../images/radtrax_01_column_retrieval/curve_of_growth.svg
:label: fig-rx1-cog
:alt: Plume optical depth against column enhancement from 0 to 1 kilogram per square metre. The linear model for band 12 is a straight line reaching 0.89. The exact band 12 curve bends below it, reaching 0.57. The exact band 11 curve is about a tenth of band 12. A dotted line marks the plume's peak at 0.3.

The curve of growth for band 12 and band 11, on the synthetic spectrum.
:::

The exact optical depth falls short of the linear one by 3.8 % at
0.05 kg/m², 15 % at the plume's peak (0.3 kg/m², a 23 % dip in band 12), 22 %
at 0.5 and 36 % at 1 kg/m². Band 11 absorbs about a tenth as much, which is
the ratio this spectrum was built with.

:::{warning} Differs from the plan
The source note says the relationship between $\tau_{\text{obs}}$ and
$\Delta\Omega$ "can exhibit mild non-linearities caused by instrument
line-shape smoothing", and defines $\sigma_{\text{CH}_4}(\lambda)$ as the
SRF-weighted average of the line-by-line cross-section. The averaging is
the cause: the band averages $e^{-m\sigma\Omega}$, not $\sigma$, and
$\langle e^{-m\sigma\Omega} \rangle \ne e^{-m\langle\sigma\rangle\Omega}$
once line cores are optically thick. The background is already in those
cores, so the deficit depends on $\Omega_{\text{bg}}$ and on $m$ as well as
on $\Delta\Omega$. That is why the table in stage 4 is indexed by $m$ and
built on top of the background.
:::

(rx1-stage-3)=
### Stage 3 · The multispectral proxy baseline (geotoolz, radtrax)

**TL;DR.** Estimate the plume-free band 12 from band 11 two ways, invert
linearly, and integrate the plume's mass.

**Problem.** Under the plume, $\rho_{12} \tau_{\text{bg},12}$ is unknown.
Band 11 sees the same surface. (a) The source note fits
$R_{12} \approx k R_{11} + b$ on pixels outside a suspected-plume mask and
uses $k R_{11} + b$ as the baseline. (b) Varon et al.'s multi-band
single-pass (MBSP) retrieval [@varon2021] fits $R_{11} \approx c R_{12}$
through the origin over the whole scene and uses the fractional change

$$
\Delta R = \frac{c R_{12} - R_{11}}{R_{11}} , \qquad
-\ln(1 + \Delta R) = m \, (\bar\sigma_{12} - \bar\sigma_{11}) \, \Delta\Omega ,
$$ (rx1-eq-mbsp)

to first order, which keeps band 11's own absorption. Both are scored by the
plume's mass above the median of a ring of clean pixels around it, as the
integrated mass enhancement method does.

```{code-block} python
:caption: Two proxy baselines on the Sentinel-2 scene, from a real run.
:linenos:

# Shapes: H = W = 200; plume mask 3,777 pixels; true mass 108,979 kg
# (a) R₁₂ ≈ k R₁₁ + b on clean pixels;  τ = −ln(R₁₂ / (k R₁₁ + b))
k, b = np.polyfit(R11[~suspect], R12[~suspect], 1)  # 0.816, 0.0016  user
omega_a = -np.log(R12 / (k * R11 + b)) / (m * s12)  # B11 "blind"    user
# → 80 % of the mass
# (b) R₁₁ ≈ c R₁₂ through the origin;  ΔΩ = −ln(c R₁₂ / R₁₁) / (m (σ̄₁₂ − σ̄₁₁))
c = (R11 * R12).sum() / (R12 * R12).sum()           # 1.220          user
omega_b = -np.log(c * R12 / R11) / (m * (s12 - s11))                  # user
# → 89 % of the mass; noise off the plume 0.029 (a), 0.032 (b) kg/m²
gz.plume.SBMP()(scene, reference_scene)   # the multi-pass ratio      today
```

The note's recipe loses a fifth of the plume. Two causes add: band 11
absorbs under the plume, so the baseline is too low there by about
$\bar\sigma_{11} / \bar\sigma_{12}$ of the signal, a tenth here; and
saturation (stage 2) takes most of the rest. MBSP removes the first by
subtracting $\bar\sigma_{11}$, and recovers 89 %. Neither baseline can
model the second surface cover: its patches, where band 12 is 0.80 of band
11 rather than 0.84, come out as false enhancements of 0.04 kg/m²
([](#fig-rx1-retrievals)). That surface term, not the sensor noise, sets
the noise floor.

:::{warning} Differs from the plan
The note calls band 11 "blind to methane". Sentinel-2's band 11 spans 1557
to 1668 nm in its response table and overlaps methane's 1.66 μm band, so a
plume darkens it too; Varon et al. account for this, and the note's linear
inversion with $\bar\sigma_{12}$ alone does not. The note's regression, with
an intercept and on clean pixels only, is a variant of MBSP rather than MBSP
itself, which regresses through the origin over the whole scene and maps
$\Delta R$ to $\Delta\Omega$ with a radiative transfer model. geotoolz ships
the **multi-pass** ratio, against a reference scene, as `plume.SBMP`; it has
no single-pass proxy.
:::

(rx1-stage-4)=
### Stage 4 · A look-up table of the band observable (radtrax)

**TL;DR.** Tabulate the MBSP observable exactly over air-mass factor and
enhancement, and invert each pixel by interpolation.

**Problem.** Replace the first-order relation in {eq}`rx1-eq-mbsp` by the
exact band observable

$$
g(m, \Delta\Omega) =
-\ln \frac{T_{12}(m, \Omega_{\text{bg}} + \Delta\Omega)}{T_{12}(m, \Omega_{\text{bg}})}
+ \ln \frac{T_{11}(m, \Omega_{\text{bg}} + \Delta\Omega)}{T_{11}(m, \Omega_{\text{bg}})} ,
$$ (rx1-eq-g)

tabulated once, and solve $g(m, \Delta\Omega) = -\ln(c R_{12} / R_{11})$ for
$\Delta\Omega$ in each pixel.

```{code-block} python
:caption: Building and inverting the table, from a real run.
:linenos:

# Shapes: table (7, 81) over m ∈ [2, 3.5] and ΔΩ ∈ [0, 0.4] kg/m²
# g(m, ΔΩ) from the exact band transmittances, both bands
lut = radtrax.Table(observable="mbsp", sensor="S2A",
                    m=np.linspace(2, 3.5, 7),
                    omega=np.linspace(0, 0.4, 81))                 proposed (radtrax)
# ΔΩ = g⁻¹(−ln(c R₁₂ / R₁₁); m), odd-extended for negative noise
omega_lut = lut.invert(-np.log(c * R12 / R11), m)  # (H, W)    proposed (radtrax)
# → 96 % of the mass; RMSE over the plume 0.0175 kg/m² (linear: 0.0206)
```

:::{figure} ../../images/radtrax_01_column_retrieval/retrievals.svg
:label: fig-rx1-retrievals
:alt: Three maps of the column enhancement. The truth shows an elongated plume peaking at 0.3. The source note's baseline with a linear inversion shows the plume fainter, with two patches of false enhancement at the top and bottom where the second surface cover is. The MBSP observable inverted with the look-up table shows the plume close to the truth, with the same two surface patches.

The truth, the source note's recipe and MBSP with the table.
:::

The table recovers 96 % of the mass. The remaining 4 % comes from the
ring's median, which the surface patches pull below zero. Because $g$ is
increasing in $\Delta\Omega$ at every $m$ in the table, the inversion is a
one-dimensional interpolation. The note's table has water vapour as a
second index. That matters where water and methane lines overlap in band
12; this synthetic spectrum has no water, so it does not test it.

(rx1-stage-5)=
### Stage 5 · Hyperspectral: a continuum against a matched filter (georeader, geotoolz)

**TL;DR.** On an EMIT-like cube, fit the note's quadratic continuum and
compare it with geotoolz's matched filter, over a surface with a carbonate
absorption.

**Problem.** A 120 × 120 cube with 68 bands from 2000 to 2500 nm, 7.4 nm
apart with an 8.5 nm full width at half maximum. Surfaces follow a quadratic
in wavelength, and 15 % of pixels carry calcite's absorption near 2340 nm,
inside the methane band. The plume peaks at 0.15 kg/m². (a) The note's
continuum: a quadratic per pixel through the 2120–2150 and 2430–2470 nm
windows, then a least-squares column over the 27 bands from 2200 to
2400 nm. (b) The matched filter [@thompson2015], with the linearised target
$t = -\mu \odot m \bar\sigma$ for the background mean $\mu$:

$$
\Delta\hat\Omega = \frac{(x - \mu)^\top \Sigma^{-1} t}{t^\top \Sigma^{-1} t} .
$$ (rx1-eq-mf)

```{code-block} python
:caption: The continuum fit and geotoolz's matched filter on the cube, from a real run.
:linenos:

# Shapes: cube (C, H, W) = (68, 120, 120); 9 window bands, 27 fit bands
srf_h = reflectance.srf(centres, fwhm, WLH)         # (G, C) Gaussian   today
# (a) R / (a λ² + b λ + c) fitted on the windows; ΔΩ by least squares
omega_cont = continuum_column(cube, windows, fit)   # (H, W)            user
# (b) Δ̂Ω = (x − μ)ᵀ Σ⁻¹ t / tᵀ Σ⁻¹ t,  t = −μ ⊙ m σ̄
mf = gz.matched_filter.MatchedFilter(target=t, fit_on_call=True, axis=0)  # today
omega_mf = mf(cube)                 # background includes the plume     today
masked = cube.copy(); masked[:, suspect] = np.nan   # NaN: left out of the fit
mf_bg = gz.matched_filter.MatchedFilter(target=t, axis=0).fit(masked)    # today
omega_mf_bg = mf_bg.transform(cube)                                     # today
# mass: continuum 95 %, matched filter 72 % (plume in Σ), 91 % (masked)
# over carbonate: continuum +0.0069, matched filter +0.0002 kg/m²
```

:::{figure} ../../images/radtrax_01_column_retrieval/hyperspectral.svg
:label: fig-rx1-hyperspectral
:alt: Three maps of the column enhancement on the hyperspectral cube. The truth shows a plume peaking at 0.15. The continuum fit shows the plume with much speckle. The matched filter with a masked background shows the plume with little speckle.

The truth, the note's continuum fit and the matched filter with the plume
left out of its background.
:::

The continuum fit recovers 95 % of the mass but reads carbonate as methane:
pixels with calcite come out 0.0069 kg/m² high, against 0.0029 kg/m² low
over the rest, and its noise off the plume is 0.0095 kg/m². The matched
filter learns the carbonate from the scene covariance and leaves no bias
there (0.0002 kg/m²), with a noise of 0.0016 kg/m². Its one trap is the
plume itself: with the plume inside the background statistics, the filter
recovers only 72 % of the mass; with the suspected plume masked out of the
fit, 91 %. The rest is saturation, as in stage 2, because the target is
linearised. At the peak, 0.15 kg/m² mixed over a 500 m layer is a 442 ppm
enhancement (xrtoolz's `column_mass_to_delta_vmr`).

:::{warning} Differs from the plan
The note presents continuum fitting as the hyperspectral strategy. EMIT's
methane enhancement product and the airborne and spaceborne imaging
spectrometers it follows use a matched filter, after Thompson et al.
[@thompson2015]; geotoolz ships one with shrinkage covariances and
cluster-wise backgrounds. A continuum fit stays useful as a check and where
too few background pixels exist for a covariance. Whether the note's windows
are free of methane and water lines on a real line list is
[open](#rx1-open-questions); on this spectrum they are, by construction.
:::

(rx1-radtrax)=
## What radtrax would own

:::{table} Each step of the retrieval: what the stack and plumax do today, and what radtrax would hold.
:label: tbl-rx1-owners

| Step | In the stack today | In plumax today | radtrax (proposed) |
|---|---|---|---|
| read reflectance, offsets, angles | georeader: Sentinel-2 L1C, EMIT, PRISMA, EnMAP readers | | |
| response functions | georeader: `read_srf` (Sentinel-2 table), `srf` (Gaussian), `transform_to_srf` | `radtran.srf.SpectralResponseFunction` | one band-integration operator, reading georeader's tables through `[geo]` |
| cross-sections from a line list | | `hapi_lut`: HITRAN through HAPI, per gas, cached | moved from plumax |
| band transmittance $T_b(m, \Omega)$ | | `radtran.forward`: exact, Maclaurin and Taylor Beer–Lambert | moved, with the background column of {eq}`rx1-eq-exact` |
| baselines | geotoolz: `plume.SBMP` (multi-pass), `spectral.ContinuumRemoval` | | `SBMP` moved from geotoolz, and the single-pass MBSP observable {eq}`rx1-eq-mbsp` |
| table and inversion | | `radtran.nb_lut`: a normalised-brightness table | `Table` over $(m, \Delta\Omega, \text{H}_2\text{O})$, monotone inversion |
| matched filter | geotoolz: `MatchedFilter`, `ColumnEnhancement` | `matched_filter`, `radtran.matched_filter` | one core, merged from the three |
| variational retrieval | | `assimilation`: 3D-Var on the radiance operator | moved, or split with vardax ([open](../package_boundaries.md#pb-open-questions)) |
| units, mass and rate | xrtoolz: `column_mass_to_delta_vmr`; geotoolz: `ColumnToMass`, `IMEEstimate` | | the plume operators go to plumax instead |
:::

Written in JAX, the band transmittance is differentiable, so the table
becomes optional: a Gauss–Newton step on {eq}`rx1-eq-exact` inverts each
pixel directly, as [Report 17](../../reports/17_cycle.md)'s Laplace step
does on the five-band sensor, and a learned retrieval can be trained
through it ([Report 5](../../reports/05_pipekit_jax.md)).

(rx1-open-questions)=
## Open questions

1. **Are the note's continuum windows clean?** 2120–2150 and 2430–2470 nm
   need checking against HITRAN's methane and water lines at EMIT's
   resolution; the long-wave window sits at the edge of methane's 2.3 μm
   band.
2. **Which band 11 to band 12 ratio does Sentinel-2 really have?** Stage 3's
   bias scales with $\bar\sigma_{11} / \bar\sigma_{12}$; the tenth used here
   is a choice, and the real ratio follows from HITRAN and the two response
   functions.
3. **A table or a solver?** A table is fast and auditable; a differentiable
   forward model is exact at every pixel and composes with learning. radtrax
   could ship both, the table built from the forward model.
4. **Which atmosphere?** The single layer with a geometric air-mass factor
   ignores scattering and the vertical profile. When does a layered model
   pay for itself on Sentinel-2's bands?

(rx1-todos)=
## Upcoming work

:::{table} Follow-ups from checking the source note.
:label: tbl-rx1-todos

| Follow-up | Where | Why |
|---|---|---|
| Rerun the example on HITRAN lines | this page's notebook, with a cached line list | the spectrum here is synthetic |
| Check `plume.SBMP`'s name and citation | geotoolz `plume/_src/operators.py:118` | it computes a two-band ratio against a reference scene, which Varon et al. [@varon2021] call multi-band multi-pass; its docstring's reference should be checked against that paper |
| Add the single-pass MBSP observable | radtrax (or geotoolz `plume` until radtrax exists) | stage 3: the stack has no single-pass proxy |
| Ship a Landsat L1 reader path with the MTL scaling and angle bands | georeader | the note's Landsat route does not exist in the stack |
:::

(rx1-references)=
## References

- The source note on the forward model, provenance, proxies and look-up
  table [@geostack_radtrax_note].
- Varon et al. on Sentinel-2 methane retrievals [@varon2021], and on the
  integrated mass enhancement [@varon2018].
- Thompson et al. on the matched filter for imaging spectroscopy
  [@thompson2015].
- georeader-spaceml, geotoolz and xrtoolz at the revisions in the header.
