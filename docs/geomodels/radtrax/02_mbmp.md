---
title: The SBMP, MBSP and MBMP family
subtitle: radtrax · 2 · Multi-band and multi-pass methane retrievals, from the radiative transfer equation to the double difference
short_title: radtrax · 2 · MBMP
description: Two derivations of the Sentinel-2 single-band multi-pass, multi-band single-pass and multi-band multi-pass methane retrievals, merged and checked against geotoolz's SBMP operator, plumax's band table and exact band transmittances on a synthetic two-pass scene. The chain from the radiative transfer equation holds; the scale factor's fitting set, the linear inversion, the band-11 cross-section and the stack's naming do not.
keywords:
  - radtrax
  - MBMP
  - MBSP
  - SBMP
  - Sentinel-2
  - methane retrieval
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

Sentinel-2 sees methane as a dip in band 12 (2.2 μm), but so does a darker
patch of ground. The single-band multi-pass (SBMP), multi-band single-pass
(MBSP) and multi-band multi-pass (MBMP) retrievals remove the ground in three
ways: compare two dates, compare two bands, or both. This page merges two
derivations of the family, from the radiative transfer equation to the
double difference, and runs every method on one synthetic two-pass scene
with the nuisances the derivations name. A stable mineral patch, a wetted
field, a patch whose spectral ratio changes, a 5 % scene-wide brightening
and a 1 % water-vapour change are all in it. The chain holds: MBMP cancels
everything but the patch whose band ratio changed, which reads as
0.039 kg/m² of methane. Four corrections come out of the run:

- **Fit the scale factor on background pixels.** Fitted over every pixel, as
  one derivation does, the plume moves it by 0.31 % and MBMP loses 11 % of
  the plume's mass.
- **Invert with the band table, not a slope.** A linear inversion in
  $m\,\Delta\alpha$ recovers 91 % of the mass; the exact two-band table
  99 %.
- **Band 11 absorbs.** Setting $\sigma_{11} = 0$ and using $\bar\sigma_{12}$
  as the slope, as the other derivation does, recovers 78 %.
- **geotoolz's `SBMP` is an MBMP.** It takes two bands and two passes, and
  without per-pass scale factors it reads the 1 % water-vapour change as a
  0.013 kg/m² offset.

+++

:::{note} About this page
:class: dropdown

| | |
|---|---|
| **Status** | Design. The two-pass retrieval ships as geotoolz's `plume.SBMP` and the band table as plumax's `radtran.nb_lut`; the [package boundaries](../package_boundaries.md) page moves both into radtrax. Nothing here changes code |
| **Reading time** | about 25 minutes |
| **Audience** | anyone running a Sentinel-2 or Landsat methane retrieval, and anyone choosing between the single-pass and multi-pass forms |
| **Companions** | [radtrax · 1](01_column_retrieval.md) (the single-pass column retrieval, the curve of growth and the look-up table this page reuses); [plumax · 1](../plumax/01_emission_rate.md) (the emission rate from the column map) |
| **Code checked against** | geotoolz 0.10.2 at `088475e` and georeader-spaceml 2.4.1 from `references/`, plumax 0.1.2 at `28f257e` (a clone; not a submodule yet). Read on 14 October 2026 |
| **Running example** | every number and figure is from one run of `notebooks/geomodels/radtrax_02_mbmp.py`, which asserts each one. The methane spectrum is radtrax · 1's synthetic one ($\bar\sigma_{11} = 0.04$, $\bar\sigma_{12} = 0.40$ m²/kg, a modelling choice); the Sentinel-2A band 11 and 12 response functions are real |
| **Sources** | two derivations supplied by the author: "Deriving the MBMP method family from first principles" [@geostack_mbmp_note] and a second, step-by-step optical derivation (generated with Gemini) [@geostack_mbmp_note_gemini]; Varon et al. [@varon2021; @varon2018], Chen et al. [@chen2022], Watine-Guiu et al. [@watineguiu2023] |
:::

:::{important} What changed from the two derivations
Both derivations run the same chain: drop emission and scattering from the
radiative transfer equation, integrate to Beer–Lambert, average over the
band, then remove the surface by ratios across passes, bands, or both. That
chain **stands**, and the two agree on the algebra up to the direction of
the scale factor $c$. Where they differ from each other, or from the run:

- **The scale factor.** Derivation (1) fits $c$ over every pixel, plume
  included; derivation (2) over background pixels. Background is right: the
  plume biases an all-pixel $c$ by 0.31 %, a uniform offset that costs MBMP
  11 % of the mass.
- **Band 11.** Derivation (2) sets $\sigma_{11} = 0$. Band 11 absorbs weakly,
  so the sensitivity is $\alpha_{12} - \alpha_{11}$, as derivation (1) has
  it; with $\bar\sigma_{12}$ alone and a Taylor expansion, MBMP recovers
  78 % of the mass.
- **The linearisation.** Both invert a straight line. The plume at 0.3 kg/m²
  is past the linear range: the slope overstates band 12's optical depth by
  16 % there. Derivation (1)'s validity condition, small
  $\Delta\Omega / \Omega_{\text{bg}}$, is the wrong one: the plume is 27
  times the background column, and what matters is the curvature of
  $\ln \mathcal{T}_b$.
- **The sensitivity.** Derivation (1) defines
  $\alpha_b = -\partial \ln \mathcal{T}_b / \partial \Omega$ and then
  multiplies it by $m$ again; the definition that matches its model is
  $\alpha_b = -m^{-1}\, \partial \ln \mathcal{T}_b / \partial \Omega$.
- **Difference of logs, not of δ.** Subtracting $\delta_{\text{MBSP}}$
  between passes cancels the surface only to first order; subtracting
  $\ln(1 + \delta)$ cancels it exactly, and is what geotoolz computes.
- **Names in the stack.** geotoolz's `plume.SBMP`, given a reference scene,
  computes the MBMP log signal without the scale factors.

Smaller points: derivation (1)'s TOA intensity drops the $\cos\theta_s$ that
derivation (2) keeps; its formulas' `\,` thin spaces and `\!` arrived as
commas and `!`, and its subscripts were eaten by Markdown italics
(`\mathcal{T}*b`), all restored here; its summary chain says MBSP cancels
"temporal A via regression", where it cancels the albedo within one pass.
A **Differs** box marks each place.
:::

(rx2-rte)=
## The radiative transfer equation in the SWIR

**TL;DR.** Two terms of the radiative transfer equation vanish in the
shortwave infrared, and what is left integrates to Beer–Lambert.

**Problem.** For the intensity $I_\lambda$ in a plane-parallel atmosphere,

$$
\mu \frac{dI_\lambda}{d\tau_\lambda} = I_\lambda - J_\lambda ,
$$ (rx2-eq-rte)

with $\mu = \cos\theta$, $\tau_\lambda$ the optical depth and $J_\lambda$ the
source function, emission plus scattering into the beam. Between 1.6 and
2.3 μm both are small. Thermal emission: a 300 K surface emits $2\times10^{-4}$
of the sunlight it reflects at 2.2 μm (albedo 0.2, $\theta_s = 35°$).
Rayleigh scattering: its optical depth falls as $\lambda^{-4}$, 16 times
weaker at 2 μm than at 1 μm, and is $3.7\times10^{-4}$ in band 12. With
$J_\lambda \to 0$, along the path $s$,

$$
\frac{dI_\lambda}{ds} = -\sigma_\lambda\, n(s)\, I_\lambda ,
$$ (rx2-eq-extinction)

with $\sigma_\lambda$ the absorption cross-section and $n$ the absorber's
density. Integrated down to a Lambertian surface of albedo $A_\lambda$ and
back up, through the column $\Omega = \int n\, dz$ along the slant paths, it
gives the TOA reflectance

$$
R_\lambda = \frac{\pi L_\lambda}{E_{\lambda,\odot} \cos\theta_s}
= A_\lambda\, \exp(-m\, \sigma_\lambda\, \Omega) , \qquad
m = \frac{1}{\cos\theta_s} + \frac{1}{\cos\theta_v} ,
$$ (rx2-eq-beer)

where $L_\lambda$ is the radiance at the sensor, $E_{\lambda,\odot}$ the
solar irradiance and $m$ the geometric air-mass factor. Every pixel has two
unknowns per wavelength, $A_\lambda$ and $\Omega$; the rest of the page is
about removing $A_\lambda$.

```{code-block} python
:caption: The two dropped terms, in numbers, from a real run.
:linenos:

# Shapes: scalars; λ in μm
# B_λ(T) = 2hc² / λ⁵ / (exp(hc / λkT) − 1)
planck(2.2, 300.0) / L_reflected      # → 2.0 × 10⁻⁴           user
# τ_R(λ) ≈ 0.008569 λ⁻⁴ (1 + 0.0113 λ⁻² + 0.00013 λ⁻⁴)
tau_rayleigh(1.0) / tau_rayleigh(2.0) # → 16.1                 user
tau_rayleigh(2.2)                     # → 3.7 × 10⁻⁴           user
```

:::{warning} Differs from the plan
Derivation (1) writes the TOA intensity as $I_\odot A_\lambda / \pi$ and
drops the $\cos\theta_s$ of the incoming irradiance; derivation (2) keeps it,
and the reflectance {eq}`rx2-eq-beer` divides it out. Neither treats
aerosols, which scatter more than air at 2.2 μm over hazy or dusty scenes;
the ratio methods below remove a scene-wide aerosol change for the same
reason they remove a scene-wide brightening, but not a local one.
:::

(rx2-band)=
## The band model and its linearisation

**TL;DR.** Average over the band, split the column into background and
plume, and see how far a straight line carries.

**Problem.** Band $b$ integrates over its spectral response $f_b(\lambda)$.
With the albedo flat across the band,

$$
R_b = A_b\, \mathcal{T}_b(\Omega) , \qquad
\mathcal{T}_b(\Omega) = \frac{\int f_b(\lambda)\, e^{-m \sigma_\lambda \Omega}\, d\lambda}
{\int f_b(\lambda)\, d\lambda} .
$$ (rx2-eq-band)

Split $\Omega = \Omega_{\text{bg}} + \Delta\Omega$, with
$\Omega_{\text{bg}} = 0.0109$ kg/m² (1.9 ppm). The exact plume optical depth
of band $b$ is
$\tau_b(\Delta\Omega) = -\ln[\mathcal{T}_b(\Omega_{\text{bg}} + \Delta\Omega) / \mathcal{T}_b(\Omega_{\text{bg}})]$,
and its tangent at the background is the sensitivity

$$
\alpha_b = -\frac{1}{m} \frac{\partial \ln \mathcal{T}_b}{\partial \Omega} \bigg|_{\Omega_{\text{bg}}} ,
\qquad
R_b \approx \underbrace{A_b\, \mathcal{T}_b(\Omega_{\text{bg}})}_{R_b^0}\,
e^{-m\, \alpha_b\, \Delta\Omega} .
$$ (rx2-eq-alpha)

```{code-block} python
:caption: The band sensitivities and the linear range, from a real run.
:linenos:

# Shapes: G = 100,000 wavelengths (0.01 nm); m = 2.225
f11, f12 = read_srf("S2A")[["B11", "B12"]]                 # today
# σ̄_b = ⟨σ⟩_b: 0.040 and 0.400 m²/kg (chosen)
# α_b = −(1/m) ∂ ln T_b / ∂Ω at Ω_bg       → 0.0398, 0.3953  user
# m α₁₂ ΔΩ / τ₁₂(ΔΩ) at ΔΩ = 0.3 kg/m²     → 1.16           user
# nB_b(ΔX) = ⟨exp(−m σ ΔX)⟩_b, no background         (181,)
nb = build_nb_lut(ds, srf, T_K=280, p_atm=0.8, amf=m)       # today
# −ln nB₁₂ / τ₁₂ at 0.3 kg/m²                → 1.011          user
```

$\alpha_b$ sits just below $\bar\sigma_b$, because the background has
already darkened the strongest line cores. The straight line holds near the
background and not at a strong plume: at 0.3 kg/m², $m\,\alpha_{12}\,\Delta\Omega$
is 16 % above the exact optical depth, the curve of growth of
[radtrax · 1](01_column_retrieval.md). Band 11 is the 1.66 μm methane band's
edge and absorbs a tenth as much as band 12 on this spectrum (the ratio is a
modelling choice here; it is not zero on real spectra either). plumax's band
table, `nB`, integrates the plume alone, without the background column; on
this spectrum that overstates band 12's plume optical depth by 1.1 %.

:::{warning} Differs from the plan
Derivation (1) defines $\alpha_b = -\partial \ln \mathcal{T}_b / \partial \Omega$,
which already contains $m$, and then writes $e^{-\alpha_b m \Delta\Omega}$; the
$1/m$ in {eq}`rx2-eq-alpha` makes the two consistent. It calls the
linearisation valid for small $\Delta\Omega / \Omega_{\text{bg}}$; a 0.3 kg/m²
plume is 27 times $\Omega_{\text{bg}}$, and the condition that matters is the
curvature of $\ln \mathcal{T}_b$ over the plume's range. Derivation (2) goes
one step further, $e^{-x} \approx 1 - x$, which needs $m \sigma \Delta\Omega \ll 1$;
at the plume peak it is 0.27.
:::

(rx2-methods)=
## Three ways to remove the surface

**TL;DR.** Ratio across passes (SBMP), across bands (MBSP), or both (MBMP);
each ratio cancels what it assumes is constant.

**Problem.** Write {eq}`rx2-eq-alpha` for bands 11 and 12, on a plume pass
$(p)$ and a plume-free reference pass $(r)$ over the same pixels. The three
estimators are:

$$
\begin{aligned}
\text{SBMP:}\quad & s = -\ln \frac{c_s R_{12}^{(p)}}{R_{12}^{(r)}} = \tau_{12}(\Delta\Omega) + \ln\frac{c_s A_{12}^{(p)}}{A_{12}^{(r)}} , \\
\text{MBSP:}\quad & s = -\ln(1 + \delta^{(p)}) , \quad
\delta = \frac{c\, R_{12}}{R_{11}} - 1
= \tau_{12} - \tau_{11} + \ln\frac{c\, A_{12}\, \mathcal{T}_{12}^{\text{bg}}}{A_{11}\, \mathcal{T}_{11}^{\text{bg}}} , \\
\text{MBMP:}\quad & s = -\ln \frac{1 + \delta^{(p)}}{1 + \delta^{(r)}}
= \tau_{12} - \tau_{11} + \ln\frac{c^{(p)} r^{(p)}}{c^{(r)} r^{(r)}} ,
\quad r = \frac{A_{12}}{A_{11}} .
\end{aligned}
$$ (rx2-eq-three)

Each scale factor is a least-squares fit through the origin over background
pixels: $c_s$ from $R_{12}^{(r)} \approx c_s R_{12}^{(p)}$, and $c$ per pass from
$R_{11} \approx c\, R_{12}$. The leftover log is what each method assumes
away. SBMP needs the band-12 albedo unchanged between passes, up to one
scene-wide factor. MBSP needs the band ratio $r$ the same everywhere in the
scene, up to $c$. MBMP needs only $r$ unchanged at each pixel between
passes, up to one scene-wide factor per pass: it is a difference in
differences, across bands and then across dates. The column comes from
inverting the exact table, $\Delta\Omega = \tau_{12}^{-1}(s)$ for SBMP and
$\Delta\Omega = g^{-1}(s)$ with $g = \tau_{12} - \tau_{11}$ for MBSP and MBMP,
or, to first order, $s / (m\, \alpha_{12})$ and $s / (m\, \Delta\alpha)$ with
$\Delta\alpha = \alpha_{12} - \alpha_{11}$.

```{code-block} python
:caption: The three estimators on a two-pass cube, from a real run.
:linenos:

# Shapes: obs, ref (2, 200, 200) = (B11, B12) × (H, W)
# c = Σ x y / Σ x² over background pixels                 () per fit
c_s = fit_c(R12r, R12p, bg)                                # user
c_p, c_r = fit_c(R11p, R12p, bg), fit_c(R11r, R12r, bg)    # user
# δ = c R₁₂ / R₁₁ − 1, per pass               (200, 200) → (200, 200)
d_p, d_r = c_p * R12p / R11p - 1, c_r * R12r / R11r - 1    # user
# ΔΩ = τ₁₂⁻¹(−ln(c_s R₁₂ / R₁₂′))              (200, 200) → (200, 200)
sbmp = invert(-np.log(c_s * R12p / R12r), tau12_tab)       # user
# ΔΩ = g⁻¹(−ln(1 + δ))                          (200, 200) → (200, 200)
mbsp = invert(-np.log1p(d_p), g_mb)                        # user
# ΔΩ = g⁻¹(−ln((1 + δ) / (1 + δ′)))             (200, 200) → (200, 200)
mbmp = invert(-(np.log1p(d_p) - np.log1p(d_r)), g_mb)      # user
# ln(R₁₁/R₁₂) − ln(R₁₁′/R₁₂′), no scale factors   (2, 200, 200) → ()
gz = gz.plume.SBMP(swir1=0, swir2=1, axis=0)(obs, ref)     # today
```

:::{warning} Differs from the plan
The two derivations write $c$ in opposite directions: derivation (1)
divides, $\delta = R_{12} / (c\, R_{11}) - 1$ with $c \approx R_{12}/R_{11}$, and
derivation (2) multiplies, $\delta = c\, R_{12} / R_{11} - 1$ with
$c \approx R_{11}/R_{12}$; they are the same estimator, and this page uses
derivation (2)'s form. Derivation (1)'s SBMP assumes
$A_{12}^{(p)} = A_{12}^{(r)}$ exactly and fits no factor; the published SBMP
[@varon2021] fits one between the passes, $c_s$ above. Both derivations
subtract $\delta^{(p)} - \delta^{(r)}$; with $1 + \delta = e^{-x}(1 + \varepsilon)$,
that difference is $(1 + \varepsilon)(e^{-x} - 1)$, so the surface term
survives as a multiplier on the signal. The log of the ratio,
{eq}`rx2-eq-three`, removes it.
:::

(rx2-running)=
## The running example: one scene, every nuisance

**TL;DR.** One plume, two passes, and every surface and atmospheric change
the derivations warn about, each in its own patch.

**Problem.** A 200 × 200 scene with 20 m pixels holds a plume of peak
0.3 kg/m² (148,700 kg) on the plume pass only. Both passes share a textured
surface: albedo varies by 15 %, the band ratio $r$ by 4 %, and a mineral
patch has $r$ 15 % low. Between the passes the whole scene brightens 5 %,
a field is wetted (both bands × 0.8), one patch's ratio drops 3 %, and
band 12's water-vapour transmission drops 1 % everywhere. Pixels enter
through plumax's `inject_plume`, with the band table built against the
background column.

:::{figure} ../../images/radtrax_02_mbmp/methods.svg
:label: fig-rx2-methods
:alt: Six maps of the column enhancement. The truth shows one plume. The source note's SBMP shows the plume weakened, a uniform negative offset and the wetted field as strong false methane. The scaled SBMP removes most of the offset but keeps the wetted field. MBSP is dominated by the surface texture and the mineral patch. MBMP shows the plume alone, with a faint patch where the spectral ratio changed. geotoolz's SBMP shows the plume, the same faint patch, and a uniform positive offset.

The truth and five retrievals on the same two-pass scene. Outlines: the
stable mineral patch (brown), the wetted field (blue), the patch whose
spectral ratio changed (purple).
:::

:::{table} What each retrieval makes of the scene: the share of the plume's mass it recovers, its offset over clean pixels, and each patch's mean above the offset (kg/m²).
:label: tbl-rx2-results

| Retrieval | Mass | Offset | Mineral | Wetted | Ratio change | Clean sd |
|---|---|---|---|---|---|---|
| SBMP, no scale factor (derivation 1) | 0.46 | −0.045 | 0.000 | **0.28** | 0.036 | 0.000 |
| SBMP, scene-wide $c_s$ | 0.86 | −0.011 | 0.000 | **0.29** | 0.035 | 0.000 |
| MBSP | 0.76 | −0.016 | **0.21** | 0.006 | 0.023 | **0.054** |
| **MBMP**, exact table, $c$ on background | **0.99** | −0.001 | 0.000 | 0.000 | 0.039 | 0.000 |
| MBMP, $c$ over every pixel (derivation 1) | 0.89 | −0.009 | 0.000 | 0.000 | 0.039 | 0.000 |
| MBMP, log ratio, linear in $m\,\Delta\alpha$ | 0.91 | −0.001 | 0.000 | 0.000 | 0.039 | 0.000 |
| MBMP, $\delta$ difference, linear in $m\,\Delta\alpha$ (derivation 1) | 0.88 | −0.001 | 0.000 | 0.000 | 0.039 | 0.000 |
| MBMP, $\delta$ difference, $\sigma_{11} = 0$, $\bar\sigma_{12}$ (derivation 2) | 0.78 | −0.001 | 0.000 | 0.000 | 0.035 | 0.000 |
| geotoolz `plume.SBMP` (with a reference), exact table | 1.16 | **+0.013** | 0.000 | 0.000 | 0.040 | 0.000 |
:::

The hierarchy is as the derivations say. SBMP, with or without a scale
factor, reads the wetted field as 0.28–0.29 kg/m² of methane, nearly the
plume's peak; MBSP reads the stable mineral patch as 0.21 kg/m² and the 4 %
ratio texture as 0.054 kg/m² of clutter; MBMP removes both, and the
scene-wide brightening and water-vapour change with them. What it keeps is
the patch whose band ratio changed: 3 % reads as 0.039 kg/m². The plume
mass then depends on two choices the derivations make differently. A scale
factor fitted over every pixel takes in the plume, moves by 0.31 % and
leaves a −0.009 kg/m² offset that costs 11 % of the mass. And the slope that
converts the signal to a column matters: the exact table recovers 99 %, a
linear inversion 91 %, the $\delta$ difference 88 %, and $\bar\sigma_{12}$ with
$\sigma_{11} = 0$ 78 %. geotoolz's `SBMP` is the MBMP log signal plus
$\ln(c^{(p)} / c^{(r)})$, a constant: it removes every surface term MBMP
removes, and keeps the 1 % water-vapour change as a +0.013 kg/m² offset over
the whole scene, which is 16 % extra mass on this plume.

:::{warning} Differs from the plan
Derivation (1)'s table says SBMP is vulnerable to temporal surface change
and MBSP to spectral slope, and MBMP to their joint change only. The run
agrees, with two refinements. A **scene-wide** change of either kind (the
brightening, the water vapour) is absorbed by the scale factors, so it is
not a failure of SBMP with $c_s$ or of MBSP; only a **local** change is. And
the scale factors are themselves fitted on the scene, so a large plume or a
large changed patch inside the fitting set moves every pixel's retrieval;
the fit should exclude a first-pass detection, as derivation (2) has it.
:::

(rx2-noise)=
## Noise

**TL;DR.** Each ratio cancels a surface term and adds the noise of the
bands it uses.

**Problem.** With independent noise of relative size $\varepsilon$ in each band
and pass, the log ratios add variances: SBMP uses two band-12 images,
MBSP one image of each band, MBMP four images, and the two-band methods
divide by the smaller $\Delta\alpha$. To first order,

$$
\text{sd}(\Delta\hat\Omega) \approx
\frac{\sqrt{2}\, \varepsilon}{m\, \alpha_{12}} \ (\text{SBMP}), \quad
\frac{\sqrt{2}\, \varepsilon}{m\, \Delta\alpha} \ (\text{MBSP}), \quad
\frac{2\, \varepsilon}{m\, \Delta\alpha} \ (\text{MBMP}) .
$$ (rx2-eq-noise)

```{code-block} python
:caption: Retrieval noise for 1 % band noise on a stable surface, from a real run.
:linenos:

# Shapes: obs, ref (2, 200, 200); ε = 0.01 per band and pass
# sd(ΔΩ̂_noisy − ΔΩ̂_clean) off the plume       (200, 200) → ()
# → SBMP 0.016, MBSP 0.018, MBMP 0.025 kg/m²; theory within 3 %
```

MBMP pays for its robustness with 1.6 times SBMP's noise. On a surface that
is stable in time, SBMP with a scale factor is the more precise retrieval;
on a surface that is uniform in its band ratio, MBSP needs no reference
pass. MBMP is the default because real surfaces are neither, and its noise
falls with averaging where a surface artefact does not.

(rx2-not-optimal)=
## When MBMP is not the right choice

Derivation (1) lists three cases, and the run bears on each:

1. **No good reference pass.** MBMP and SBMP both need a plume-free date
   with the same surface. If the only reference is from another season,
   the local ratio changes grow and MBSP, which needs none, may do better.
2. **A strongly wavelength-dependent surface.** Where $r$ varies widely
   across the scene, one $c$ fits it poorly; MBMP still cancels the stable
   part of $r$ (the mineral patch is 0.000 kg/m² above the offset), but the
   fitted $c$ is less well defined, and SBMP avoids the band ratio.
3. **Water vapour that changes between passes.** Band 12 has water-vapour
   lines. A scene-wide change is a scene-wide factor and the per-pass $c$
   absorbs it; a change that varies across the scene (a front, terrain)
   does not, and reads like a changed band ratio: 1 % is 0.013 kg/m².

(rx2-cancels)=
## What each method cancels

:::{table} The family as a difference in differences, from the run.
:label: tbl-rx2-cancels

| Method | Removes | Leaves | Noise (1 %) | Reference pass |
|---|---|---|---|---|
| SBMP | stable surface; scene-wide change (with $c_s$) | any local change in band 12 (wetted field: 0.29) | 0.016 | needed |
| MBSP | scene-wide change; the mean band ratio | the band ratio's spatial texture (0.054) and stable patches (mineral: 0.21) | 0.018 | none |
| MBMP | stable surface, in both bands; scene-wide change; local changes that keep the band ratio | local changes of the band ratio (3 %: 0.039) | 0.025 | needed |
:::

(rx2-stack)=
## Where it lives in the stack

| Piece | Today | Notes |
|---|---|---|
| Band transmittance $\mathcal{T}_b$, the tables $\tau_b$ and $g$ | plumax `radtran.nb_lut` (`build_nb_lut`, `lookup_nb`, `inject_plume`) | integrates the plume without the background column (1.1 % on band 12 here) |
| Sentinel-2 response functions | georeader `S2_SAFE_reader.read_srf` | real SRFs; B11 1569–1658 nm, B12 2113–2286 nm at half maximum |
| The two-pass, two-band log signal | geotoolz `plume.SBMP` with a reference scene | an MBMP without scale factors; the single-pass fallback is a normalised difference, not an MBSP |
| Scale factors, background masks, exact inversion | none | written in the notebook |

:::{warning} Differs from the plan
geotoolz's operator is named `SBMP` and documented as a single-band
multi-pass retrieval, but with a reference scene it uses both bands on both
passes: $\ln(R_{11}/R_{12}) - \ln(R_{11}'/R_{12}')$, which the notebook
confirms is the MBMP log signal plus a constant to $10^{-11}$. Without the
per-pass factors the constant, $\ln(c^{(p)}/c^{(r)})$, stays: on this scene the
1 % water-vapour change. Its docstring cites "Satellite discovery of
anomalously large methane point sources from oil/gas production" (RSE, 251)
for the formula; the method names and this formula are Varon et al.
2021 [@varon2021]. Its single-pass fallback,
$(R_{11} - R_{12}) / (R_{11} + R_{12})$, is a detection prefilter, as the
docstring says, not the MBSP of {eq}`rx2-eq-three`.
:::

(rx2-open-questions)=
## Open questions

1. **What should the operators be called?** An `MBMP` operator in radtrax,
   with `SBMP` and `MBSP` beside it, each with the scale factors and a
   background mask, would match the literature; the current `SBMP` could
   become MBMP's `scale_factors=False` mode.
2. **Where does the background mask come from?** The run uses a first-pass
   detection grown by a Gaussian of 2 pixels; an iterated fit, or the plume mask from
   geotoolz's `PlumeMask`, are the alternatives, and the choice moves the
   offset.
3. **Should the band table include the background?** plumax's `nB` omits
   it; including it is one extra integral and changes band 12 by 1 % here,
   more over high background or long air-mass paths.

(rx2-todos)=
## Upcoming work

:::{table} Follow-ups from merging the two derivations.
:label: tbl-rx2-todos

| Follow-up | Where | Why |
|---|---|---|
| Rename `plume.SBMP` (with a reference) to MBMP, and add per-pass scale factors | geotoolz now, radtrax after the move | stage 4: it is an MBMP, and without $c$ a scene-wide water-vapour change is a 0.013 kg/m² offset |
| Add SBMP and MBSP operators with scale factors fitted on a background mask | radtrax | stage 4: an all-pixel fit costs 11 % of the mass |
| Invert with the two-band table $g = \tau_{12} - \tau_{11}$ | radtrax, on plumax's `nb_lut` | stage 4: 99 % of the mass against 91 % for the linear slope |
| Build `nB` against the background column | plumax `radtran.nb_lut` | stage 2: 1.1 % on band 12 |
| Share the synthetic spectrum and band transmittance between radtrax · 1 and 2 | radtrax | both notebooks carry the same helpers |
:::

(rx2-references)=
## References

- The two derivations: "Deriving the MBMP method family from first
  principles" [@geostack_mbmp_note] and the step-by-step optical
  derivation [@geostack_mbmp_note_gemini].
- The SBMP, MBSP and MBMP retrievals [@varon2021]; the IME and the
  fine-scale plume retrievals they feed [@varon2018]; a ground-truth
  calibration of Sentinel-2 retrievals [@chen2022]; transient emissions
  from geostationary observations [@watineguiu2023].
- geotoolz, georeader and plumax at the revisions in the header.
