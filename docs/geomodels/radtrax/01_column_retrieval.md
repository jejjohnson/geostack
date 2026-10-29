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

A methane plume over a bright surface removes a little bit of light in the short-wave infrared, and the single-layer Beer–Lambert model is how we turn that deficit into a column mass. In this page, we write the model out term by term, we say where each term comes from in a satellite product, and we follow the two ways one can estimate the plume-free baseline: a proxy band for multispectral sensors such as Sentinel-2, and a continuum fit for hyperspectral sensors such as EMIT. We then invert the model with a look-up table. The physics of the source note holds. However, five of its details do not: 1) Sentinel-2's band 11 is not blind to methane, 2) the effective cross-section is only valid for weak absorption, 3) Sentinel-2 L1C has a radiometric offset, 4) Landsat L1 is not NetCDF or HDF5, and 5) the operational hyperspectral retrievals are matched filters, not continuum fits. On a synthetic Sentinel-2 scene, the note's recipe recovers 80 % of the plume's mass whereas the multi-band ratio with a look-up table recovers 96 %. **radtrax** (formerly rtmx) is the GeoModels package for radiative transfer in JAX, and it would own the band transmittance, the table and its inversion. A lot of this already exists in plumax (band-integrated radiative transfer, a HITRAN table pipeline, a matched filter), which the [package boundaries](../package_boundaries.md) page proposes to move.

+++

:::{note} About this page
:class: dropdown

| | |
|---|---|
| **Status** | Design. radtrax does not exist yet; this page fixes the physics its first module implements, and names what the stack and plumax already do |
| **Reading time** | about 25 minutes |
| **Audience** | anyone writing or reviewing a methane column retrieval, and anyone deciding what radtrax owns |
| **Companions** | [radtrax · 2](02_mbmp.md) (the multi-pass family and the MBSP estimator); [radtrax · 3](03_design.md) (radtrax's design); [plumax · 1](../plumax/01_emission_rate.md) (the emission rate from the column); the [package boundaries](../package_boundaries.md) page (where the code moves); [Report 16](../../reports/16_geo_data_stack.md#r16-running) (one methane scene through geotoolz, with the matched filter and the IME); [Report 9](../../reports/09_use_cases.md#r9-running) (the use cases built on the same scene); [Report 5](../../reports/05_pipekit_jax.md) (pipelines under JAX, for a differentiable radtrax); [Report 17](../../reports/17_cycle.md) (the five-band sensor, a simpler Beer–Lambert model) |
| **Code checked against** | georeader-spaceml 2.4.1 (the Sentinel-2 L1C reader, its SRF table COPE-GSEG-EOPG-TN-15-0007 v4.0, the EMIT reader and the Gaussian SRF helper); geotoolz 0.10.2 at `088475e` (`plume`, `spectral`, `matched_filter`); xrtoolz 0.0.5 at `308c7f2` (`atm.gas.ch4`). Read on 14 October 2026 |
| **Running example** | every number and figure is from one run of `notebooks/geomodels/radtrax_01_column_retrieval.py`, which builds the scenes from fixed seeds and asserts each number the page shows; code comments carry the [status markers](../../overview_examples.md#ex-markers) |
| **Sources** | the source note, "The Core Forward Model Equation" and its four companion sections [@geostack_radtrax_note]; Varon et al. on Sentinel-2 [@varon2021]; Thompson et al. on the matched filter [@thompson2015] |
:::

:::{important} The spectrum is synthetic
HITRAN is not reachable from the environment that ran the example, so we draw the methane lines at random in the 1.66 μm and 2.3 μm bands and scale them so that the band-averaged cross-sections are σ̄₁₂ = 0.40 m²/kg and σ̄₁₁ = 0.04 m²/kg (a ratio of one tenth, which we chose for the example). The Sentinel-2 response functions are the real ones. So every bias below has the sign and the cause it would have on a real spectrum, but its size is the size it has on this one. Rerunning the notebook on HITRAN lines is the first item of [](#tbl-rx1-todos).
:::

:::{important} What changed from the source note
The source note's **forward model and its inversion stand**. Its details do not, and each [Differs box](../../overview.md#sec-conventions) argues one:

- **Units and nuisances**: $\sigma$ per mass, $\tau_\text{bg}$ a nuisance ([variables](#rx1-variables)).
- **Sentinel-2 L1C offset, Landsat L1 format** ([sources](#rx1-provenance)).
- **The effective cross-section** is the weak-absorption limit ([stage 2](#rx1-stage-2)).
- **Band 11 is not blind to methane** ([stage 3](#rx1-stage-3)).
- **Hyperspectral products use a matched filter** ([stage 5](#rx1-stage-5)).
:::

(rx1-model)=
## The forward model

The reflectance a sensor measures in band $b$ at the pixel location $\vect{s} \in \mathbb{R}^2$ is the surface reflectance times the transmittance of the two-way path through the atmosphere, averaged over the band's spectral response $f_b(\lambda)$ on the spectral range $\mathcal{S}$:

$$
R_b(\vect{s}) = \rho_b(\vect{s}) \frac{\int_{\mathcal{S}} f_b(\lambda) \exp\left(-\mathrm{AMF}(\vect{s}) \sigma(\lambda) [\Omega_\text{bg} + \Delta\Omega(\vect{s})]\right) \mathrm{d}\lambda}{\int_{\mathcal{S}} f_b(\lambda) \mathrm{d}\lambda} .
$$ (rx1-eq-exact)

Here $\sigma(\lambda)$ is the methane cross-section per unit mass, $\Omega_\text{bg}$ is the background column and $\Delta\Omega$ is the plume's enhancement, and every symbol is a scalar (see [](#tbl-rx1-vars)). When every line is optically thin, the band average of the exponential is the exponential of the band-averaged cross-section, so {eq}`rx1-eq-exact` becomes the source note's model:

$$
R_b = \rho_b \tau_{\text{bg},b} \exp(-\mathrm{AMF} \bar{\sigma}_b \Delta\Omega) ,
$$ (rx1-eq-weak)

where the band-averaged cross-section is

$$
\bar{\sigma}_b = \frac{\int_{\mathcal{S}} f_b(\lambda) \sigma(\lambda) \mathrm{d}\lambda}{\int_{\mathcal{S}} f_b(\lambda) \mathrm{d}\lambda} .
$$ (rx1-eq-sigma-bar)

This means the plume's optical depth is linear in the column:

$$
\tau_b = -\ln \frac{R_b}{\rho_b \tau_{\text{bg},b}} = \mathrm{AMF} \bar{\sigma}_b \Delta\Omega .
$$ (rx1-eq-tau)

{eq}`rx1-eq-weak` is the right picture for small plumes and the wrong one for strong lines. The running example measures how wrong it is in [stage 2](#rx1-stage-2).

(rx1-variables)=
## The variables

[](#tbl-rx1-vars) gives each term of {eq}`rx1-eq-exact` with its shape, its units and the piece of the stack that supplies it. They are all scalars per band and per pixel, so none of them is bold; only the pixel location $\vect{s}$ and the spectra of [stage 5](#rx1-stage-5) are vectors.

:::{table} Every term of the model, where it comes from, and the stack piece that supplies it today.
:label: tbl-rx1-vars

| Symbol | Name | Role | Shape | Units | From | In the stack today |
|---|---|---|---|---|---|---|
| $R_b(\vect{s})$ | top-of-atmosphere reflectance | input | $\mathbb{R}_{\ge 0}$ | dimensionless | the L1 product | georeader's Sentinel-2 L1C and EMIT readers |
| $\rho_b(\vect{s})$ | surface reflectance | nuisance | $\mathbb{R}_{\ge 0}$ | dimensionless | scene statistics, as part of the baseline | |
| $\tau_{\text{bg},b}(\vect{s})$ | background transmittance (methane, water vapour, CO₂) | nuisance | $[0, 1]$ | dimensionless | scene statistics, or a radiative transfer model | |
| $\mathrm{AMF}(\vect{s})$ | two-way geometric air-mass factor | input | $\mathbb{R}_{\ge 2}$ | dimensionless | sun and view zenith angles in the metadata | georeader's `sza` and `vza` grids (Sentinel-2) |
| $f_b(\lambda)$ | spectral response of band $b$ | parameter | $\mathbb{R}_{\ge 0}$ | dimensionless | the agency's SRF table | georeader's `read_srf` (Sentinel-2) and `srf` (Gaussian) |
| $\sigma(\lambda)$, $\bar{\sigma}_b$ | mass absorption cross-section, and its band average | parameter | $\mathbb{R}_{>0}$ | m²/kg | a line list, then the band's response | georeader's `read_srf` (Sentinel-2) and `srf` (Gaussian); plumax's `hapi_lut` (HITRAN through HAPI) and `radtran.srf` |
| $\Omega_\text{bg}$ | background methane column | parameter | $\mathbb{R}_{>0}$ | kg/m² | climatology or a model | xrtoolz's `dry_air_column` and `mixing_ratio_to_column` |
| $\Delta\Omega(\vect{s})$ | column mass enhancement | output | $\mathbb{R}$ | kg/m² | the inversion | geotoolz's matched filter; xrtoolz's `column_mass_to_delta_vmr` converts it |
:::

On this page, $\Omega$ is the methane column (as in the source note) and not a spatial domain, and $\rho$ is the surface reflectance and not an air density.

:::{warning} Differs from the plan
The source note gives $\sigma$ in m²/kg, which is what {eq}`rx1-eq-tau` needs for a column in kg/m². However, HITRAN's cross-sections are per molecule, in cm²/molecule, so the conversion is $\sigma_\text{mass} = 10^{-4} N_A \sigma_\text{mol} / M_{\mathrm{CH_4}}$ in m²/kg (with $N_A$ the Avogadro constant and $M_{\mathrm{CH_4}}$ the molar mass). The note also lists $\tau_\text{bg}$ as a "parameter layer". It is a nuisance which the baseline absorbs, like $\rho$, and it is never estimated on its own.
:::

(rx1-provenance)=
## Where each term comes from

Each term has one source in a satellite product ([](#fig-rx1-provenance)).

:::{figure} ../../images/radtrax_01_column_retrieval/provenance.svg
:label: fig-rx1-provenance
:alt: Six rows. The satellite L1 product gives digital numbers or radiance, scaled to reflectance R_b. The metadata gives the sun and view zenith angles, combined into the air-mass factor AMF. The agency SRF table and a line list give the band cross-sections and transmittances. Scene statistics give the baseline from clean pixels or spectral windows. A radiative transfer model gives the look-up table.

The sources of the model's terms, from the notebook.
:::

**Reflectance.** For Sentinel-2 L1C, the granule's JPEG 2000 bands are top-of-atmosphere reflectance scaled to integers:

$$
R_b = \frac{\mathrm{DN}_b + \text{RADIO\_ADD\_OFFSET}_b}{\text{QUANTIFICATION\_VALUE}} ,
$$ (rx1-eq-dn)

with a quantification value of 10,000 and, since processing baseline 04.00 (January 2022), an offset of −1,000. Hyperspectral L1 products are radiance $L_b$ instead, which we convert with the band's solar irradiance $E_b$ and the Earth–Sun distance $d$ in astronomical units:

$$
R_b = \frac{\pi L_b d^2}{E_b \cos(\mathrm{SZA})} .
$$ (rx1-eq-rad)

**Air-mass factor.** The geometric two-way path through a plane-parallel layer is

$$
\mathrm{AMF} = \frac{1}{\cos(\mathrm{SZA})} + \frac{1}{\cos(\mathrm{VZA})} ,
$$ (rx1-eq-amf)

where $\mathrm{SZA}$ and $\mathrm{VZA}$ are the sun and view zenith angles.

**Cross-section.** A line list gives $\sigma(\lambda)$ at the layer's pressure and temperature. The band's response function then averages it, {eq}`rx1-eq-sigma-bar`, or, for the exact model, averages the transmittance, {eq}`rx1-eq-exact`.

```{code-block} python
:caption: Reading the terms with the stack today.
:linenos:

# Shapes: S2 L1C tile (13, H, W); EMIT L1B (285, H, W); SRF table (877, 13)
s2 = S2ImageL1C(safe_path)                                              # today
s2.quantification_value(), s2.radio_add_offsets()  # 10000, {B12: -1000}  today
# R_b = (DN_b + offset_b) / Q
R = s2.load()                                     # reflectance           today
# AMF = 1/cos(SZA) + 1/cos(VZA), from the 5 km angle grids
sza, vza = s2.sza, s2.vza["B12"]                  # GeoTensors            today
m = 1 / np.cos(np.radians(sza)) + 1 / np.cos(np.radians(vza))           # user
# R_b = π L_b d² / (E_b cos(SZA))
emit = EMITImage(l1b_path); R_h = emit.load(as_reflectance=True)        # today
srf = read_srf("S2A")              # B11, B12 at 1 nm, offline            today
```

:::{warning} Differs from the plan
There are three corrections to the note's sources. **Sentinel-2 L1C** is $(\mathrm{DN} + \text{offset}) / 10{,}000$, {eq}`rx1-eq-dn`, for products from processing baseline 04.00 onwards, not $\mathrm{DN} / 10{,}000$, and georeader reads the offset from the product metadata (`radio_add_offsets`). Its angles are 5 km grids in `MTD_TL.xml`, which are interpolated to the pixels. **Landsat L1** is GeoTIFF, with `REFLECTANCE_MULT_BAND_x` and `REFLECTANCE_ADD_BAND_x` in `MTL.txt` and a division by the sine of the sun elevation. Its `MTL.txt` only holds the scene-centre sun angles, so the per-pixel angles come from the angle bands. **EMIT and PRISMA** deliver radiance, so {eq}`rx1-eq-rad` comes first, and EMIT's geometry is in a separate observation file (which georeader reads).
:::

(rx1-running)=
## The running example: one plume, two sensors

The scene is a 4 km × 4 km Sentinel-2 scene at 20 m, so 200 × 200 pixels. The surface has a smooth band 11 reflectance between 0.20 and 0.40 and two cover types, with band 12 at 0.84 and 0.80 of band 11 and a 1 % per-pixel scatter. A Gaussian plume, elongated and rotated by 30°, peaks at $\Delta\Omega = 0.3$ kg/m². The background is 1.9 ppm (a column of 0.0109 kg/m²), and the sun and view zenith angles are 35° and 5°, so {eq}`rx1-eq-amf` gives $\mathrm{AMF} = 2.225$. Each band adds noise with a standard deviation of 0.0015 in reflectance. [Stage 5](#rx1-stage-5) then adds an EMIT-like cube.

(rx1-stage-1)=
### Stage 1 · The forward model (georeader → radtrax)

**TL;DR.** Build the exact model {eq}`rx1-eq-exact` with Sentinel-2's own response functions.

```{code-block} python
:caption: The forward model with Sentinel-2A's band 11 and band 12, from a real run.
:linenos:

# Shapes: N_λ = 100,000 wavelengths (0.01 nm); H = W = 200
srf = read_srf("S2A")                     # (877, 13) at 1 nm         today
f11, f12 = np.interp(WL, srf.index, srf["B11"]), ...      # (N_λ,)     user
# σ̄_b = ∫ f_b σ dλ / ∫ f_b dλ
s11, s12 = band_mean(sigma, f11), band_mean(sigma, f12)     # 0.04, 0.40  user
# T_b(AMF, Ω) = ∫ f_b exp(−AMF σ Ω) dλ / ∫ f_b dλ
T = radtrax.band_transmittance(sigma, f12)        # (AMF, Ω) → T   proposed (radtrax)
# R_b = ρ_b T_b(AMF, Ω_bg + ΔΩ) + ε_b
R12 = rho12 * T(m, OMEGA_BG + d_omega) + eps        # (H, W)       user
# T₁₂(AMF, Ω_bg) = 0.990: the background takes 1 % of band 12
```

Sentinel-2A's band 12 is above half its peak from 2113 to 2286 nm, and band 11 reaches 1668 nm. The background methane already removes 1 % of band 12's light along this path.

(rx1-stage-2)=
### Stage 2 · The effective cross-section against the exact band (radtrax)

**TL;DR.** Compare the linear optical depth {eq}`rx1-eq-tau` with the exact one, the curve of growth.

**Problem.** The plume's exact band optical depth, above the background, is

$$
\tau_b(\Delta\Omega) = -\ln \frac{T_b(\mathrm{AMF}, \Omega_\text{bg} + \Delta\Omega)}{T_b(\mathrm{AMF}, \Omega_\text{bg})} ,
$$ (rx1-eq-cog)

where $T_b(\mathrm{AMF}, \Omega) \in [0, 1]$ is the band transmittance of a column $\Omega$, and the linear model claims that $\tau_b = \mathrm{AMF} \bar{\sigma}_b \Delta\Omega$. However, a line's core saturates long before its wings do, so the band absorbs less than the linear model says.

:::{figure} ../../images/radtrax_01_column_retrieval/curve_of_growth.svg
:label: fig-rx1-cog
:alt: Plume optical depth against column enhancement from 0 to 1 kilogram per square metre. The linear model for band 12 is a straight line reaching 0.89. The exact band 12 curve bends below it, reaching 0.57. The exact band 11 curve is about a tenth of band 12. A dotted line marks the plume's peak at 0.3.

The curve of growth for band 12 and band 11, on the synthetic spectrum.
:::

The exact optical depth {eq}`rx1-eq-cog` falls short of the linear one ([](#fig-rx1-cog)) by 3.8 % at 0.05 kg/m², by 15 % at the plume's peak (0.3 kg/m², where band 12 dips by 20 % and not the 23 % the linear model gives), by 22 % at 0.5 and by 36 % at 1 kg/m². Band 11 absorbs about a tenth as much, which is simply the ratio we built this spectrum with.

:::{warning} Differs from the plan
The source note says the relationship between $\tau_\text{obs}$ and $\Delta\Omega$ "can exhibit mild non-linearities caused by instrument line-shape smoothing", and it defines $\sigma_{\mathrm{CH_4}}(\lambda)$ as the SRF-weighted average of the line-by-line cross-section. The averaging is actually the cause: the band averages $\exp(-\mathrm{AMF} \sigma \Omega)$ and not $\sigma$, and $\langle \exp(-\mathrm{AMF} \sigma \Omega) \rangle \ne \exp(-\mathrm{AMF} \langle \sigma \rangle \Omega)$ once the line cores are optically thick. The background is already in those cores, so the deficit depends on $\Omega_\text{bg}$ and on $\mathrm{AMF}$ as well as on $\Delta\Omega$. This is why the table in stage 4 is indexed by $\mathrm{AMF}$ and built on top of the background.
:::

(rx1-stage-3)=
### Stage 3 · The multispectral proxy baseline (geotoolz, radtrax)

**TL;DR.** Estimate the plume-free band 12 from band 11 in two ways, invert linearly, and integrate the plume's mass.

**Problem.** Under the plume, $\rho_{12} \tau_{\text{bg},12}$ is unknown, but band 11 sees the same surface. There are two ways to use it. a) The source note fits $R_{12} \approx \theta_1 R_{11} + \theta_0$ on the pixels outside a suspected-plume mask and uses $\theta_1 R_{11} + \theta_0$ as the baseline, where $\theta_0, \theta_1 \in \mathbb{R}$ are the intercept and the slope. b) The multi-band single-pass (MBSP) retrieval of Varon et al. [@varon2021], whose estimator [radtrax · 2](02_mbmp.md#rx2-methods) derives, fits $R_{11} \approx c R_{12}$ through the origin over the whole scene with one ratio $c \in \mathbb{R}$, and uses the fractional change

$$
\Delta R = \frac{c R_{12} - R_{11}}{R_{11}} ,
$$ (rx1-eq-mbsp)

which, to first order, gives the column as

$$
-\ln(1 + \Delta R) = \mathrm{AMF} (\bar{\sigma}_{12} - \bar{\sigma}_{11}) \Delta\Omega .
$$ (rx1-eq-mbsp-tau)

This keeps band 11's own absorption. We score both by the plume's mass above the median of a ring of clean pixels around it, as the [integrated mass enhancement](../plumax/01_emission_rate.md#er1-ime) method does.

```{code-block} python
:caption: Two proxy baselines on the Sentinel-2 scene, from a real run.
:linenos:

# Shapes: H = W = 200; plume mask 3,777 pixels; true mass 108,979 kg
# (a) R₁₂ ≈ θ₁ R₁₁ + θ₀ on clean pixels;  τ = −ln(R₁₂ / (θ₁ R₁₁ + θ₀))
slope, intercept = np.polyfit(R11[~suspect], R12[~suspect], 1)  # 0.816, 0.0016  user
omega_a = -np.log(R12 / (slope * R11 + intercept)) / (m * s12)  # B11 "blind"    user
# → 80 % of the mass
# (b) R₁₁ ≈ c R₁₂ through the origin;  ΔΩ = −ln(c R₁₂ / R₁₁) / (AMF (σ̄₁₂ − σ̄₁₁))
c = (R11 * R12).sum() / (R12 * R12).sum()           # 1.220          user
omega_b = -np.log(c * R12 / R11) / (m * (s12 - s11))                  # user
# → 89 % of the mass; noise off the plume 0.029 (a), 0.032 (b) kg/m²
gz.plume.SBMP()(scene, reference_scene)   # the multi-pass ratio      today
```

The note's recipe loses a fifth of the plume, and there are two causes which add up. Band 11 absorbs under the plume, so the baseline is too low there by about $\bar{\sigma}_{11} / \bar{\sigma}_{12}$ of the signal (a tenth here), and saturation (stage 2) takes most of the rest. MBSP removes the first cause by subtracting $\bar{\sigma}_{11}$, and it recovers 89 %. However, neither baseline can model the second surface cover. Its patches, where band 12 is 0.80 of band 11 rather than 0.84, come out as false enhancements of 0.04 kg/m² ([](#fig-rx1-retrievals)). So it is this surface term, not the sensor noise, that sets the noise floor.

:::{warning} Differs from the plan
The note calls band 11 "blind to methane". However, Sentinel-2's band 11 spans 1557 to 1668 nm in its response table and overlaps methane's 1.66 μm band, so a plume darkens it too. Varon et al. account for this, and the note's linear inversion with $\bar{\sigma}_{12}$ alone does not. The note's regression, with an intercept and on clean pixels only, is a variant of MBSP rather than MBSP itself, which regresses through the origin over the whole scene and maps $\Delta R$ to $\Delta\Omega$ with a radiative transfer model. geotoolz has no single-pass proxy; its `plume.SBMP` is a multi-pass retrieval ([radtrax · 2](02_mbmp.md#rx2-stack)).
:::

(rx1-stage-4)=
### Stage 4 · A look-up table of the band observable (radtrax)

**TL;DR.** Tabulate the MBSP observable exactly over the air-mass factor and the enhancement, and invert each pixel by interpolation.

**Problem.** We replace the first-order relation in {eq}`rx1-eq-mbsp-tau` by the exact band observable

$$
g(\mathrm{AMF}, \Delta\Omega) = -\ln \frac{T_{12}(\mathrm{AMF}, \Omega_\text{bg} + \Delta\Omega)}{T_{12}(\mathrm{AMF}, \Omega_\text{bg})} + \ln \frac{T_{11}(\mathrm{AMF}, \Omega_\text{bg} + \Delta\Omega)}{T_{11}(\mathrm{AMF}, \Omega_\text{bg})} ,
$$ (rx1-eq-g)

which we tabulate once, and then we solve $g(\mathrm{AMF}, \Delta\Omega) = -\ln(c R_{12} / R_{11})$ for $\Delta\Omega$ in each pixel.

```{code-block} python
:caption: Building and inverting the table, from a real run.
:linenos:

# Shapes: table (7, 81) over AMF ∈ [2, 3.5] and ΔΩ ∈ [0, 0.4] kg/m²
# g(AMF, ΔΩ) from the exact band transmittances, both bands
lut = radtrax.Table(observable="mbsp", sensor="S2A",
                    m=np.linspace(2, 3.5, 7),
                    omega=np.linspace(0, 0.4, 81))                 proposed (radtrax)
# ΔΩ = g⁻¹(−ln(c R₁₂ / R₁₁); AMF), odd-extended for negative noise
omega_lut = lut.invert(-np.log(c * R12 / R11), m)  # (H, W)    proposed (radtrax)
# → 96 % of the mass; RMSE over the plume 0.0175 kg/m² (linear: 0.0206)
```

:::{figure} ../../images/radtrax_01_column_retrieval/retrievals.svg
:label: fig-rx1-retrievals
:alt: Three maps of the column enhancement. The truth shows an elongated plume peaking at 0.3. The source note's baseline with a linear inversion shows the plume fainter, with two patches of false enhancement at the top and bottom where the second surface cover is. The MBSP observable inverted with the look-up table shows the plume close to the truth, with the same two surface patches.

The truth, the source note's recipe and MBSP with the table.
:::

The table recovers 96 % of the mass. The remaining 4 % comes from the ring's median, which the surface patches pull below zero. Because $g$ of {eq}`rx1-eq-g` is increasing in $\Delta\Omega$ at every $\mathrm{AMF}$ in the table, the inversion is a one-dimensional interpolation. The note's table has water vapour as a second index. That matters where water and methane lines overlap in band 12, but this synthetic spectrum has no water, so it does not test it.

(rx1-stage-5)=
### Stage 5 · Hyperspectral: a continuum against a matched filter (georeader, geotoolz)

**TL;DR.** On an EMIT-like cube, fit the note's quadratic continuum and compare it with geotoolz's matched filter, over a surface with a carbonate absorption.

**Problem.** The cube is 120 × 120 pixels with $N_{\mathcal{S}_y} = 68$ bands from 2000 to 2500 nm, 7.4 nm apart with an 8.5 nm full width at half maximum. The surfaces follow a quadratic in wavelength, and 15 % of the pixels carry calcite's absorption near 2340 nm, which is inside the methane band. The plume peaks at 0.15 kg/m². Again we compare two approaches. a) The note's continuum is a quadratic per pixel through the windows from 2120 to 2150 nm and from 2430 to 2470 nm, followed by a least-squares column over the 27 bands from 2200 to 2400 nm. b) The matched filter [@thompson2015] estimates the enhancement at pixel $n$ from its spectrum $\vect{y}_n \in \mathbb{R}^{N_{\mathcal{S}_y}}$ as

$$
\widehat{\Delta\Omega}_n = \frac{\vect{J}_h^\top \gmat{\Sigma}_y^{-1} (\vect{y}_n - \gvect{\mu}_y)}{\vect{J}_h^\top \gmat{\Sigma}_y^{-1} \vect{J}_h} ,
$$ (rx1-eq-mf)

where $\gvect{\mu}_y \in \mathbb{R}^{N_{\mathcal{S}_y}}$ and $\gmat{\Sigma}_y \in \mathbb{R}^{N_{\mathcal{S}_y} \times N_{\mathcal{S}_y}}$ are the background mean and covariance of the spectra, and $\vect{J}_h = -\mathrm{AMF} \gvect{\mu}_y \odot \bar{\gvect{\sigma}} \in \mathbb{R}^{N_{\mathcal{S}_y}}$ is the linearised target (the Jacobian of {eq}`rx1-eq-weak` with respect to $\Delta\Omega$ at $\Delta\Omega = 0$, with $\bar{\gvect{\sigma}}$ the band-averaged cross-sections of all the bands). The estimate $\widehat{\Delta\Omega}_n \in \mathbb{R}$ is a scalar per pixel.

```{code-block} python
:caption: The continuum fit and geotoolz's matched filter on the cube, from a real run.
:linenos:

# Shapes: cube (N_Sy, H, W) = (68, 120, 120); 9 window bands, 27 fit bands
srf_h = reflectance.srf(centres, fwhm, WLH)       # (N_λ, N_Sy) Gaussian today
# (a) R / (θ₂ λ² + θ₁ λ + θ₀) fitted on the windows; ΔΩ by least squares
omega_cont = continuum_column(cube, windows, fit)   # (H, W)            user
# (b) ΔΩ̂ = J_hᵀ Σ_y⁻¹ (y − μ_y) / (J_hᵀ Σ_y⁻¹ J_h),  J_h = −AMF μ_y ⊙ σ̄
mf = gz.matched_filter.MatchedFilter(target=j_h, fit_on_call=True, axis=0)  # today
omega_mf = mf(cube)                 # background includes the plume     today
masked = cube.copy(); masked[:, suspect] = np.nan   # NaN: left out of the fit
mf_bg = gz.matched_filter.MatchedFilter(target=j_h, axis=0).fit(masked)    # today
omega_mf_bg = mf_bg.transform(cube)                                     # today
# mass: continuum 95 %, matched filter 72 % (plume in Σ_y), 91 % (masked)
# over carbonate: continuum +0.0069, matched filter +0.0002 kg/m²
```

:::{figure} ../../images/radtrax_01_column_retrieval/hyperspectral.svg
:label: fig-rx1-hyperspectral
:alt: Three maps of the column enhancement on the hyperspectral cube. The truth shows a plume peaking at 0.15. The continuum fit shows the plume with much speckle. The matched filter with a masked background shows the plume with little speckle.

The truth, the note's continuum fit and the matched filter with the plume left out of its background.
:::

The continuum fit recovers 95 % of the mass, but it reads carbonate as methane ([](#fig-rx1-hyperspectral)): the pixels with calcite come out 0.0069 kg/m² high, against 0.0029 kg/m² low over the rest, and its noise off the plume is 0.0095 kg/m². The matched filter {eq}`rx1-eq-mf` learns the carbonate from the scene covariance $\gmat{\Sigma}_y$ and leaves no bias there (0.0002 kg/m²), with a noise of 0.0016 kg/m². Its one trap is the plume itself. With the plume inside the background statistics, the filter only recovers 72 % of the mass, whereas with the suspected plume masked out of the fit it recovers 91 %. The rest is saturation, as in stage 2, because the target is linearised. At the peak, 0.15 kg/m² mixed over a 500 m layer is a 442 ppm enhancement (xrtoolz's `column_mass_to_delta_vmr`).

:::{warning} Differs from the plan
The note presents continuum fitting as the hyperspectral strategy. However, EMIT's methane enhancement product, and the airborne and spaceborne imaging spectrometers it follows, use a matched filter after Thompson et al. [@thompson2015]. geotoolz ships one with shrinkage covariances and cluster-wise backgrounds, which [Report 16](../../reports/16_geo_data_stack.md#r16-stage-2) runs chip by chip. A continuum fit stays useful as a check, and where too few background pixels exist for a covariance. Whether the note's windows are free of methane and water lines on a real line list is still [open](#rx1-open-questions); on this spectrum they are, by construction.
:::

(rx1-radtrax)=
## What radtrax would own

Where each piece lives today is in the [package boundaries inventory](../package_boundaries.md#pb-today), and [](#tbl-rx1-owners) keeps what radtrax would hold.

:::{table} Each step of the retrieval and what radtrax would hold.
:label: tbl-rx1-owners

| Step | radtrax (proposed) |
|---|---|
| read reflectance, offsets, angles | stays in georeader |
| response functions | one band-integration operator, reading georeader's tables through `[geo]` |
| cross-sections from a line list | moved from plumax `hapi_lut` |
| band transmittance $T_b(\mathrm{AMF}, \Omega)$ | moved from plumax `radtran.forward`, with the background column of {eq}`rx1-eq-exact` |
| baselines | `SBMP` moved from geotoolz, and the single-pass MBSP observable {eq}`rx1-eq-mbsp` |
| table and inversion | `Table` over $(\mathrm{AMF}, \Delta\Omega, \mathrm{H_2O})$, monotone inversion, in place of plumax's `radtran.nb_lut` |
| matched filter | one core, merged from the three |
| variational retrieval | moved from plumax `assimilation`, or split with vardax ([open](../package_boundaries.md#pb-open-questions)) |
| units, mass and rate | stay in xrtoolz; the plume operators go to plumax |
:::

Written in JAX, the band transmittance is differentiable, so the table becomes optional. A Gauss–Newton step on {eq}`rx1-eq-exact` inverts each pixel directly, as the Laplace step of [Report 17](../../reports/17_cycle.md) does on the five-band sensor, and one can train a learned retrieval through it ([Report 5](../../reports/05_pipekit_jax.md)).

(rx1-open-questions)=
## Open questions

1. **Are the note's continuum windows clean?** The windows from 2120 to 2150 nm and from 2430 to 2470 nm need checking against HITRAN's methane and water lines at EMIT's resolution, and the long-wave window sits at the edge of methane's 2.3 μm band.
2. **Which band 11 to band 12 ratio does Sentinel-2 really have?** The bias of stage 3 scales with $\bar{\sigma}_{11} / \bar{\sigma}_{12}$. The tenth we use here is a choice, and the real ratio follows from HITRAN and the two response functions.
3. **A table or a solver?** A table is fast and auditable, whereas a differentiable forward model is exact at every pixel and composes with learning. radtrax could ship both, with the table built from the forward model.
4. **Which atmosphere?** The single layer with a geometric air-mass factor ignores scattering and the vertical profile. When does a layered model pay for itself on Sentinel-2's bands?

(rx1-todos)=
## Upcoming work

:::{table} Follow-ups from checking the source note.
:label: tbl-rx1-todos

| Follow-up | Where | Why |
|---|---|---|
| Rerun the example on HITRAN lines | this page's notebook, with a cached line list | the spectrum here is synthetic |
| Add the single-pass MBSP observable | radtrax (or geotoolz `plume` until radtrax exists) | stage 3: the stack has no single-pass proxy |
| Ship a Landsat L1 reader path with the MTL scaling and angle bands | georeader | the note's Landsat route does not exist in the stack |
:::

(rx1-references)=
## References

- The source note on the forward model, provenance, proxies and look-up table [@geostack_radtrax_note].
- Varon et al. on Sentinel-2 methane retrievals [@varon2021], and on the integrated mass enhancement [@varon2018].
- Thompson et al. on the matched filter for imaging spectroscopy [@thompson2015].
- georeader-spaceml, geotoolz and xrtoolz at the revisions in the header.
