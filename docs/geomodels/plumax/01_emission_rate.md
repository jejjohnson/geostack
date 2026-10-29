---
title: Emission rate from first principles
subtitle: plumax · 1 · The Gaussian plume, the Gaussian puff, the cross-sectional flux, the divergence method and the IME as integrals of one advection–diffusion equation
short_title: plumax · 1 · Emission rate
description: Five ways to estimate a methane point source's emission rate from a column map, each derived from the advection–diffusion equation. The source note is checked against plumax's Gaussian plume, geotoolz's cross-sectional flux and IME operators and exact integrals on a synthetic plume. The derivations hold; the IME's boundary argument, the reason the cross-sectional flux needs an effective wind and the dispersion parameterisation do not.
keywords:
  - plumax
  - emission rate
  - Gaussian plume
  - Gaussian puff
  - cross-sectional flux
  - divergence method
  - integrated mass enhancement
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

A satellite sees a methane plume as a map of column enhancement, whereas what the operator of a site needs is the emission rate $Q$. The five methods in use (the Gaussian plume, the Gaussian puff, the cross-sectional flux, the divergence method and the integrated mass enhancement, IME) are all integrals of one advection–diffusion equation, and in this page we derive each of them from it in turn. On a synthetic plume from plumax, every derivation checks out against exact integrals and against geotoolz's operators, with three corrections to the source note:

- **The IME's boundary terms are not small.** On a contour of constant column, the advective flux vanishes exactly and crosswind diffusion carries all of $Q$ out, which is why the IME's effective wind absorbs diffusion.
- **The cross-sectional flux needs an effective wind because of shear**, not diffusion. Here the concentration-weighted wind runs from 2.55 to 3.62 m/s for a 3 m/s wind at 10 m.
- **plumax's dispersion grows like $x$ near the source** (Briggs), and not like $\sqrt{x}$ as the note's constant-diffusivity formula says.

We also find that an IME calibration is tied to its definition of length. The same plume needs $U_\text{eff} = 1.17$ m/s with $L = \sqrt{A}$, and 4.37 m/s with geotoolz's default chord length.

+++

:::{note} About this page
:class: dropdown

| | |
|---|---|
| **Status** | Design. The methods ship in plumax (plume, puff, LES) and geotoolz (cross-sectional flux, IME); the [package boundaries](../package_boundaries.md) page moves geotoolz's into plumax. Nothing here changes code |
| **Reading time** | about 30 minutes |
| **Audience** | anyone turning a methane column map into an emission rate, and anyone choosing which of the five methods a product uses |
| **Companions** | [radtrax · 1](../radtrax/01_column_retrieval.md) (the column map these methods start from); [radtrax · 2](../radtrax/02_mbmp.md) (the Sentinel-2 multi-pass columns); [Report 16](../../reports/16_geo_data_stack.md#r16-stage-5) (one methane scene through geotoolz's IME); [Report 9](../../reports/09_use_cases.md#r9-running) (the use cases built on the IME); [package boundaries](../package_boundaries.md#pb-moves) (the plume operators' move to plumax) |
| **Code checked against** | plumax 0.1.2 at `28f257e`, geotoolz 0.10.2 at `088475e` and georeader-spaceml 2.4.1 and xrtoolz 0.0.5 at `308c7f2`, from `references/`. Read on 14 October 2026 |
| **Running example** | every number and figure is from one run of `notebooks/geomodels/plumax_01_emission_rate.py`, on a synthetic plume of $Q = 2$ kg/s (7.2 t/h) in a 3 m/s wind, which asserts each number the page shows; code comments carry the [status markers](../../overview_examples.md#ex-markers) |
| **Sources** | the source note [@geostack_plumax_note]; Varon et al. [@varon2018; @varon2021], Frankenberg et al. [@frankenberg2016], Krings et al. [@krings2011], Beirle et al. [@beirle2019], Liu et al. [@liu2021], Koene et al. [@koene2024], Jacob et al. [@jacob2022], Jia et al. [@jia2025] |
:::

:::{important} What changed from the source note
The note's **five derivations stand**: each method is an integral of the column-integrated advection–diffusion equation, and the cross-sectional flux and the divergence method are one statement through Gauss's theorem. What **does not** stand is argued in a [Differs box](../../overview.md#sec-conventions) each:

- **Dispersion widths grow like $x$**, and the column ignores the release height ([plume](#er1-plume)).
- **The puff sum is the plume only at high Péclet number** ([puff](#er1-puff)).
- **The cross-sectional flux's effective wind comes from shear** ([cross-sectional flux](#er1-csf)).
- **The divergence method needs the enhancement and the right box** ([divergence](#er1-divergence)).
- **The IME's boundary terms are zero and $Q$**, and its calibration is tied to its length ([IME](#er1-ime)).

The `\,` thin spaces of the formulas arrived as commas in the source (`(u, c)`, `λ, c`), and we restore them here.
:::

:::{note} Checked against a second derivation
:class: dropdown

We read a second derivation of the same five methods (generated with Gemini and supplied by the author) against this page. It contributed four steps which the source note skips, and which are now on the page and checked in the notebook: 1) the plume's source as an initial condition $U c(0^+) = Q \delta \delta$ ([plume](#er1-plume)), 2) the puff's moving-frame Green's function ([puff](#er1-puff)), 3) the step function behind the cross-sectional flux ([cross-sectional flux](#er1-csf)), and 4) the diffusive boundary term of the divergence method ([divergence](#er1-divergence)), plus the scaling argument for the IME ([IME](#er1-ime)). Where it differs from this page:

- **Its puff has no ground.** The unbounded Green's function puts half the mass below the surface for a ground release, whereas plumax's `puff_concentration`, {eq}`er1-eq-puff3d`, adds the image at $z = -H$, and the notebook checks that it conserves $M$ above ground.
- **It drops diffusion from the divergence method everywhere.** That only holds where the boundary lies outside the plume except at the downwind edge; on a box narrower than the plume, diffusion carries 1.18 of the 2 kg/s.
- **Its IME scaling balances the source against advection**, which is zero through a threshold contour ({eq}`er1-eq-adv0`). The diffusive balance gives the same $U \mathrm{IME} / L$, which is why the form survives.
- **It hangs the cross-sectional flux and the IME on the Gaussian plume.** Both of them, and the divergence method, only need the column equation, so they hold for puffs, for non-Gaussian plumes and for any $\sigma_y(x)$.
:::

(er1-equation)=
## The advection–diffusion equation

The concentration $c(\vect{s}, t)$ of a trace gas obeys

$$
\frac{\partial c}{\partial t} + \nabla \cdot (\vect{u} c) = \nabla \cdot (\mat{K} \nabla c) - \lambda c + S(\vect{s}, t) ,
$$ (er1-eq-adr)

where $\vect{s} = (x, y, z) \in \mathbb{R}^3$ is the position (along-wind, crosswind and height), $\vect{u} \in \mathbb{R}^3$ is the wind [m/s], $\mat{K} \in \mathbb{R}^{3 \times 3}$ is the turbulent diffusivity [m²/s], $\lambda$ is a first-order loss rate [1/s] and $S$ is the source. Every other symbol on this page is a scalar in $\mathbb{R}$, apart from the column flux $\vect{F}$ and the normal $\vect{n}$ of [the divergence method](#er1-divergence). For incompressible flow ($\nabla \cdot \vect{u} = 0$), the advection term is $\vect{u} \cdot \nabla c$.

**Methane.** Methane has a lifetime of about nine years, so $\lambda \approx 0$ on a plume's timescale of minutes to hours. We align $x$ with a mean wind $(U, 0, 0)$, keep a diagonal $\mat{K}$ with entries $K_x, K_y, K_z$, and drop the along-wind diffusion when the Péclet number $\mathrm{Pe} = U L / K_x$ is large:

$$
\frac{\partial c}{\partial t} + U \frac{\partial c}{\partial x} = K_y \frac{\partial^2 c}{\partial y^2} + K_z \frac{\partial^2 c}{\partial z^2} + S .
$$ (er1-eq-highpe)

**The column.** A satellite measures the column $\Omega(x, y, t) = \int_0^\infty c \mathrm{d}z$. With no flux through the ground and $c \to 0$ aloft, the vertical diffusion integrates to zero, and for a point source $q = Q \delta(x) \delta(y)$ we get

$$
\frac{\partial \Omega}{\partial t} + U \frac{\partial \Omega}{\partial x} = K_y \frac{\partial^2 \Omega}{\partial y^2} + Q \delta(x) \delta(y) .
$$ (er1-eq-column)

Pulling $U$ out of the $z$ integral assumes that the wind does not vary with height, and [stage 2](#er1-csf) measures what that costs. Each method below comes either from {eq}`er1-eq-highpe` or from this column equation ([](#fig-er1-derivation)).

:::{figure} ../../images/plumax_01_emission_rate/derivation.svg
:label: fig-er1-derivation
:alt: The advection-diffusion equation at the top. The Gaussian plume (steady, solved) and the Gaussian puff (Green's function) come from it directly. Its column integral, the column equation, gives the cross-sectional flux (an integral over y), the divergence method (the horizontal divergence of the column flux, ∇_h · F) and the IME (an area integral, then a residence time τ_res).

The five methods and the equation each comes from.
:::

(er1-plume)=
## The Gaussian plume (plumax)

**TL;DR.** Solve {eq}`er1-eq-highpe` at steady state, integrate it over height, and invert the column for $Q$.

**Problem.** At steady state, $x$ plays the role of time (the travel time $\tau_x = x / U$), and {eq}`er1-eq-highpe` is a two-dimensional heat equation in $(y, z)$, $\partial_{\tau_x} c = K_y \partial_{yy} c + K_z \partial_{zz} c$. Integrating {eq}`er1-eq-highpe` across $x = 0$ turns the source into its initial condition, $U c(0^+, y, z) = Q \delta(y) \delta(z - H)$, so the source injects $Q / U$ per metre of travel. With a reflecting ground and a release height $H$,

$$
c(x, y, z) = \frac{Q}{2\pi U \sigma_y \sigma_z} e^{-y^2 / 2\sigma_y^2} \left[ e^{-(z-H)^2 / 2\sigma_z^2} + e^{-(z+H)^2 / 2\sigma_z^2} \right] ,
$$ (er1-eq-plume3d)

where $\sigma_y(x)$ and $\sigma_z(x)$ are the crosswind and vertical dispersion widths. The column of {eq}`er1-eq-plume3d` is

$$
\Omega(x, y) = \frac{Q}{\sqrt{2\pi} U \sigma_y(x)} e^{-y^2 / 2\sigma_y^2(x)} ,
$$ (er1-eq-plumecol)

which we invert for the emission rate on the plume's axis:

$$
Q = \sqrt{2\pi} U \sigma_y(x) \Omega(x, 0) .
$$ (er1-eq-plumeq)

```{code-block} python
:caption: plumax's 3D plume, integrated over height, against the column formula, from a real run.
:linenos:

# Shapes: Q = 2 kg/s, U = 3 m/s, class D; x = 500, 1500 m
# σ = a x (1 + b x)^c, Briggs, for y and z
D = get_dispersion_params("D")                         # today
# c(x, y, z) by (er1-eq-plume3d), with the image source
c = plume_concentration(x, y, z, 0, 0, H, U, 0, Q, D)  # today
# Ω = ∫₀^∞ c dz, quadrature, H = 0 and 30 m          (·) → ()
# → Q / (√(2π) U σ_y) e^{−y²/2σ_y²} to 10⁻⁶         user
# σ_y: 39.0 m at 500 m, 146.1 m at 2 km
# d ln σ_y / d ln x = 0.99 between 100 and 400 m
# U ∬ c dy dz on a 1 m grid, H = 30 m              (Y, Z) → ()
# → Q to 10⁻³ at 500 m and 1.5 km                   user
```

The column formula {eq}`er1-eq-plumecol` holds for any release height. The two image terms together integrate to $\int_{-\infty}^{\infty} e^{-(z-H)^2/2\sigma_z^2} \mathrm{d}z = \sqrt{2\pi} \sigma_z$ whatever $H$ is, so plumax's plume released at 30 m has the same column as one released at the ground, to quadrature accuracy. In addition, the flux through every cross-section, $U \iint c \mathrm{d}y \mathrm{d}z$, is $Q$, which is simply the initial condition carried downwind (and the cross-sectional flux in three dimensions).

:::{warning} Differs from the plan
The note derives $\sigma_y^2 = 2 K_y x / U$, which is the constant-diffusivity solution, and then lists the Pasquill–Gifford stability classes as the way to parameterise $\sigma_y(x)$. However, the two disagree: plumax's Briggs coefficients, $\sigma_y = a x (1 + b x)^{c}$, grow like $x$ near the source (an exponent of 0.99 here, which is Taylor's short-time limit) and only like $\sqrt{x}$ far downwind, where the constant-$K$ form applies. The note's restriction to a ground-level source for the column is also unnecessary, since the column is the same for every $H$.
:::

(er1-puff)=
## The Gaussian puff (plumax)

**TL;DR.** Release mass in puffs, keep along-wind diffusion, and recover the plume by superposition.

**Problem.** An instantaneous release of mass $M$ at $(0, 0, H)$ and $t = 0$ is the initial condition $c(\vect{s}, 0) = M \delta(x) \delta(y) \delta(z - H)$ of {eq}`er1-eq-adr`, with $S = 0$ afterwards. In the frame moving with the wind, $x' = x - U t$, the chain rule cancels the advection term and leaves the heat equation $\partial_t c = K_x \partial_{x'x'} c + K_y \partial_{yy} c + K_z \partial_{zz} c$. Its Green's function is the product of three one-dimensional heat kernels, $(4\pi K_i t)^{-1/2} e^{-x_i^2 / 4 K_i t}$ (with $x_i$ the $i$-th coordinate of the moving frame), and with $\sigma_i^2 = 2 K_i t$ and an image at $z = -H$ for a reflecting ground, we get

$$
c(x, y, z, t) = \frac{M}{(2\pi)^{3/2} \sigma_x \sigma_y \sigma_z} e^{-(x - U t)^2 / 2\sigma_x^2 - y^2 / 2\sigma_y^2} \left[ e^{-(z-H)^2 / 2\sigma_z^2} + e^{-(z+H)^2 / 2\sigma_z^2} \right] ,
$$ (er1-eq-puff3d)

which is plumax's `puff_concentration`, and its column is

$$
\Omega(x, y, t) = \frac{M}{2\pi \sigma_x(t) \sigma_y(t)} \exp\left( -\frac{(x - U t)^2}{2\sigma_x^2} - \frac{y^2}{2\sigma_y^2} \right) ,
$$ (er1-eq-puffcol)

again for any release height. A steady release is then the superposition $\Omega = \int_0^t Q \Omega_1(x, y, t - t') \mathrm{d}t'$ of unit puffs, where each $\Omega_1$ is {eq}`er1-eq-puffcol` with $M = 1$.

```{code-block} python
:caption: Puffs superposed against the steady plume, K = 10 m²/s, from a real run.
:linenos:

# Shapes: release ages a ∈ (0, 3600 s], 20,000 of them
# Ω(x, 0) = ∫₀^T Q Ω₁(x, 0; a) da                    (a,) → ()
# against Q / (√(2π) U σ_y), σ_y² = 2 K x / U
superposed(500, u=5) / kplume(500, u=5)  # Pe 250 → 0.999  user
superposed(50, u=5) / kplume(50, u=5)    # Pe 25  → 0.990  user
superposed(500, u=1) / kplume(500, u=1)  # Pe 50  → 0.995  user
superposed(50, u=1) / kplume(50, u=1)    # Pe 5   → 0.958  user
# plumax's puff, σ² = 2 K t, t = 100 s, H = 30 m, 2 m grid
c = puff_concentration(x - U*t, y, z, 0, 0, H, σ, σ, σ, M)   # today
# ∭_{z ≥ 0} c dV = M to 10⁻⁶                  (301, 301, 201) → ()
# ∫ c dz = (er1-eq-puffcol) to 10⁻⁶      (301, 301, 201) → (301, 301)
# ∂c/∂t + U ∂c/∂x − K ∇²c = 0 to 10⁻⁴, central differences   user
```

The puff sum and the plume agree to 0.1 % at a Péclet number of 250, and they differ by 4 % at 5, where the along-wind diffusion (which the plume drops) matters. So this is the note's point about low winds made quantitative: the plume formula is the high-Péclet limit of the puff sum, not its exact steady state.

:::{warning} Differs from the plan
The note says that the superposition "recovers the steady-state Gaussian plume" for a constant $Q$ and $t \to \infty$. With $K_x$ retained, it recovers the steady solution **with** along-wind diffusion, which only equals the Gaussian plume when $\mathrm{Pe} \gg 1$; the difference is the 4 % above at $\mathrm{Pe} = 5$.
:::

(er1-csf)=
## The cross-sectional flux (geotoolz)

**TL;DR.** Integrate the column across the plume at several downwind distances and multiply by the wind.

**Problem.** We integrate {eq}`er1-eq-column` at steady state over all $y$. The diffusion term integrates to $K_y [\partial_y \Omega]_{-\infty}^{\infty} = 0$, because the plume and its gradient vanish far to the sides, and the source term integrates to $Q \delta(x)$:

$$
\frac{\mathrm{d}}{\mathrm{d}x} \left[ U \int_{-\infty}^{\infty} \Omega \mathrm{d}y \right] = Q \delta(x) ,
$$ (er1-eq-csfstep)

which, with nothing upwind, integrates to

$$
U \int_{-\infty}^{\infty} \Omega \mathrm{d}y = Q \Theta(x) ,
$$ (er1-eq-csfint)

where $\Theta$ is the step function. So the flux is constant downwind and equals $Q$:

$$
Q = U \int_{-\infty}^{\infty} \Omega(x, y) \mathrm{d}y \text{ for every } x > 0 .
$$ (er1-eq-csf)

When the wind $u(z)$ varies with height, the flux through a transect is $\iint u(z) c \mathrm{d}y \mathrm{d}z$, so the wind that multiplies the column is the concentration-weighted one,

$$
U_\text{eff}(x) = \frac{\int u(z) c(x, 0, z) \mathrm{d}z}{\int c(x, 0, z) \mathrm{d}z} .
$$ (er1-eq-ueffcsf)

```{code-block} python
:caption: geotoolz's cross-sectional flux on the synthetic column, from a real run.
:linenos:

# Shapes: 100 × 160 pixels at 20 m; transects every 400 m
csf = gz.plume.CrossSectionalFlux(
    source=(0, 0), wind_u=3.0, wind_v=0.0,
    n_transects=5, transect_spacing_m=400)             # today
# Q_k = U Σ_j Ω_kj Δy                          (100, 160) → (5,)
csf(column, x > 0)["flux_kg_s"]           # → 2.000 ×5  today
csf(column, column > 0.002)["flux_kg_s"]  # thresholded today
# → 1.90, 1.62, 1.25, 0.77, 0
# u(z) = 3 (z / 10 m)^0.15;  U_eff = ∫ u c dz / ∫ c dz    user
# → 2.55 m/s at 100 m, 3.38 at 1 km, 3.62 at 2 km
```

:::{figure} ../../images/plumax_01_emission_rate/csf.svg
:label: fig-er1-csf
:alt: Left, the column Ω of the synthetic plume with a 2 grams per square metre threshold contour and dotted transects every 400 metres. Right, the cross-sectional flux estimate of Q against transect distance: the full transects give 2 kilograms per second at every distance; the thresholded ones fall from 1.9 to 0 by 2 kilometres.

The cross-sectional flux on full and on thresholded transects.
:::

On full transects the flux is $Q$ at every distance, as {eq}`er1-eq-csf` says ([](#fig-er1-csf)). However, a detection threshold removes the plume's flanks, and the far transects lose more: 5 % at 400 m, 61 % at 1.6 km, and everything at 2 km, where the whole transect is below 2 g/m². Shear changes the wind that applies, so a plume hugging the ground moves slower than the 10 m wind (2.55 against 3 m/s at 100 m) whereas a deeper one moves faster (3.62 at 2 km).

:::{warning} Differs from the plan
The note introduces $U_\text{eff}$ in the cross-sectional flux "to account for the fact that the true transport is not purely advective". However, in the column equation the crosswind diffusion integrates to zero, so the true wind gives $Q$ exactly. What the effective wind accounts for is the vertical shear, {eq}`er1-eq-ueffcsf`, and the turbulent fluctuations of the wind, whose correlation with the concentration carries flux too. The truncation by a detection threshold, which the note lists, is a separate low bias, and it grows downwind.
:::

(er1-divergence)=
## The divergence method (xrtoolz budgets)

**TL;DR.** Map the source as the divergence of the column flux, and check Gauss's theorem on a box.

**Problem.** We write the column equation in flux form with the column flux $\vect{F} = \int \vect{u}_\text{h} c \mathrm{d}z \approx \vect{u}_\text{h} \Omega$, where $\vect{u}_\text{h}, \vect{F} \in \mathbb{R}^2$ are the horizontal wind and flux, and integrate it over a region $\mathcal{A}$ which holds the source. Gauss's theorem turns both divergences into boundary integrals, with $\vect{n} \in \mathbb{R}^2$ the outward unit normal and $l$ the arc length along $\partial \mathcal{A}$:

$$
Q = \oint_{\partial \mathcal{A}} \vect{F} \cdot \vect{n} \mathrm{d}l - \oint_{\partial \mathcal{A}} K \nabla_\text{h} \Omega \cdot \vect{n} \mathrm{d}l .
$$ (er1-eq-divfull)

Averaged over a long period, and dropping the diffusive term, the source density is

$$
\bar{q}(x, y) = \nabla_\text{h} \cdot \bar{\vect{F}} ,
$$ (er1-eq-div)

and the emission rate of the region is

$$
Q_{\mathcal{A}} = \iint_{\mathcal{A}} \bar{q} \mathrm{d}A = \oint_{\partial \mathcal{A}} \bar{\vect{F}} \cdot \vect{n} \mathrm{d}l .
$$ (er1-eq-divbox)

```{code-block} python
:caption: The divergence of the column flux on the grid, from a real run.
:linenos:

# Shapes: F_x = U Ω (100, 160); boxes x ∈ [−100, 1500] m
# q = ∂F_x/∂x, central differences       (100, 160) → (100, 160)
div = np.gradient(U * column, 20.0, axis=1)            # user
# Q_A = ∬_A q dA = ∮ F·n dl                      (100, 160) → ()
area_sum(box(-100, 1500, half=600))  # → 2.000 kg/s    user
area_sum(box(-100, 1500, half=60))   # → 0.81, too narrow
# the dropped term on that box, K_eff = (U/2) dσ_y²/dx (Briggs)
# 2 ∫₀^{x₁} K_eff Ω(x, h) h / σ_y² dx → 1.18; + 0.82 = Q   user
# one overpass, σ_Ω = 0.003 kg/m²: sd = U σ_Ω Δx √n_rows
# → 1.39 kg/s (2,000 draws: 1.39; this draw 1.51); 49 for a 10 % sd
```

The area sum and the boundary flux agree to $10^{-6}$, as Gauss's theorem requires, and a box wide enough to hold the plume returns $Q$. However, a box narrower than the plume returns 0.81 kg/s. The rest is the diffusive term of {eq}`er1-eq-divfull`, which the method drops: plumax's Briggs column solves the column equation with an effective diffusivity $K_\text{eff}(x) = \frac{U}{2} \mathrm{d}\sigma_y^2 / \mathrm{d}x$, and with it the diffusive flux through the narrow box's side edges is 1.18 kg/s. The advective part (0.82 exactly, 0.81 on the grid) plus the diffusive part is $Q$ to $10^{-6}$. So dropping diffusion is only safe for a boundary which crosses the plume where advection carries it (at the downwind edge) and lies outside it everywhere else. With noise, the area sum of a central difference telescopes to four edge columns, each weighted ½, so its uncertainty is theirs: 1.39 kg/s for one overpass at 0.003 kg/m² of noise (70 % of the signal), and 49 overpasses for a 10 % uncertainty. This is why the method averages many overpasses, as the note says it does at TROPOMI's scale. xrtoolz's `budgets` (`control_volume_integral`, `boundary_flux`) computes both sides of {eq}`er1-eq-divbox` on a model grid, but nothing in the stack applies them to methane columns yet.

:::{warning} Differs from the plan
There are two refinements. The column-integrated horizontal wind is not divergence free even when the 3D flow is, so $\nabla_\text{h} \cdot (\vect{u}_\text{h} \Omega)$ on the **total** column picks up $\Omega \nabla_\text{h} \cdot \vect{u}_\text{h}$ from the background. Using the **enhancement**, as the note does, is what keeps the background out, and Koene et al. [@koene2024] analyse the remainder. In addition, the divergence method and the cross-sectional flux are only identical for the same domain and the same assumptions about the side and upwind edges: on the narrow box, more than half of $Q$ leaves by the diffusive term which the method drops.
:::

(er1-ime)=
## The integrated mass enhancement (geotoolz)

**TL;DR.** Sum the mass above a detection threshold, divide by a residence time, and see what the boundary terms really are.

**Problem.** Let $\mathcal{A}$ be the region where $\Omega > \Omega_\text{thr}$, for a detection threshold $\Omega_\text{thr}$. On its boundary $\Omega = \Omega_\text{thr}$, so the advective boundary integral is

$$
\oint_{\partial\mathcal{A}} U \Omega n_x \mathrm{d}l = U \Omega_\text{thr} \oint_{\partial\mathcal{A}} n_x \mathrm{d}l = 0 ,
$$ (er1-eq-adv0)

because the outward normal of a closed curve integrates to zero. Away from the source, the total flux balance of $\mathcal{A}$ then leaves only diffusion, which means everything the source puts in leaves across the contour by crosswind diffusion,

$$
Q = -\oint_{\partial\mathcal{A}} K_y \frac{\partial \Omega}{\partial y} n_y \mathrm{d}l .
$$ (er1-eq-diffQ)

The IME sidesteps the boundary with a residence time, $\tau_\text{res} = \mathrm{IME} / Q$, which is parameterised as $\tau_\text{res} = L / U_\text{eff}$. A scaling argument gives the same form from either balance. Balancing the source against advection, $U \Omega / L \sim Q / A$ with $\mathrm{IME} \sim \Omega A$, gives $Q \sim U \mathrm{IME} / L$. Balancing it against what really leaves (the crosswind diffusion out of a plume of width $W$ and length $L$) gives $Q \sim K_y \mathrm{IME} / W^2$, and the diffusive growth $W^2 \sim K_y L / U$ turns that into the same $U \mathrm{IME} / L$. The order-one factor which the scaling leaves out is what the calibration fixes. So, with $\Omega_j$ and $A_j$ the column and the area of pixel $j$,

$$
\mathrm{IME} = \sum_{j \in \mathcal{A}} \Omega_j A_j ,
$$ (er1-eq-ime)

and the emission rate is

$$
Q = \frac{U_\text{eff} \mathrm{IME}}{L} .
$$ (er1-eq-imeq)

```{code-block} python
:caption: The contour fluxes, and the IME with each plume length, from a real run.
:linenos:

# Shapes: K plume, K_y = 10 m²/s; contour Ω = 4 g/m², 663 m
# ∮ U Ω n_x dl = −2 U Ω_thr [f(x_end) − f(0)]  → 0     user
# −∮ K_y ∂Ω/∂y n_y dl = 2 K_y Ω_thr ∫ f/σ² dx  → 2.000 = Q  user
# IME = Σ Ω_j A_j over Ω > 2 g/m², plumax: 796 kg; τ_res = 398 s
# L, chord of the mask                          (100, 160) → ()
L = plume_length(mask, transform, method="max_axis")   # today
# → 1,740 m; √A = 465 m
# U_eff = Q L / IME: 1.17 m/s (√A), 4.37 m/s (chord)    user
# Q = U L⁻¹ IME, raw wind and chord             (100, 160) → ()
ime = gz.plume.IMEEstimate(wind_speed=3.0,
                           length_method="max_axis")   # today
ime(column, mask)                         # → 1.37 kg/s
```

Both contour integrals, {eq}`er1-eq-adv0` and {eq}`er1-eq-diffQ`, are exact: the advective flux through the contour is zero, and the diffusive flux is $Q$ to $10^{-9}$. So the residence-time argument is not a way around small boundary terms. Instead, it replaces the one boundary term that carries the whole source (diffusion across the plume's edge) with a calibrated wind, which is also why $U_\text{eff}$ depends on the threshold, the noise and the turbulence, as the note says. The calibration is tied to the length scale it was fitted with: the same noise-free plume needs $U_\text{eff} = 1.17$ m/s with $L = \sqrt{A}$ and 4.37 m/s with the chord, a factor of 3.7. geotoolz's `IMEEstimate`, which computes {eq}`er1-eq-ime` and {eq}`er1-eq-imeq`, offers three length methods (`max_axis` by default, `convex_hull` and `skeleton`), and none of them is $\sqrt{A}$.

:::{warning} Differs from the plan
The note rearranges the integrated equation as $Q \approx -\iint U \partial_x\Omega \mathrm{d}A + \iint K_y \partial_{yy}\Omega \mathrm{d}A$, whereas from {eq}`er1-eq-column` it is $Q = \iint U \partial_x\Omega \mathrm{d}A - \iint K_y \partial_{yy}\Omega \mathrm{d}A$. It then calls both boundary terms small, which would make $Q$ small; in fact the advective one is zero and the diffusive one is $Q$. The note also gives $U_\text{eff} = a + b \ln U_{10}$ as the form of the calibration. However, Varon et al. fitted a logarithm for one instrument [@varon2018] and other studies fit linear forms, so the form, like the coefficients, belongs to the calibration. Lastly, because geotoolz's default length is the chord and not $\sqrt{A}$, a $U_\text{eff}$ from a $\sqrt{A}$ calibration must not be passed to it unchanged, and [Report 16's stage 5](../../reports/16_geo_data_stack.md#r16-stage-5) is such a case.
:::

(er1-summary)=
## The five methods side by side

:::{table} The five methods, what each integrates and what each needs, with where it lives in the stack.
:label: tbl-er1-summary

| Method | Integral of the equation | Assumes | Needs | Scale | In the stack today |
|---|---|---|---|---|---|
| Gaussian plume | none: solve {eq}`er1-eq-highpe` | steady state, constant $U$, high Pe | wind speed and direction, stability | point source | plumax `gauss_plume` (Briggs, NumPyro inference) |
| Gaussian puff | superposition over release times | constant $K$ per puff | wind time series, $K$ | point source | plumax `gauss_puff` (Pasquill–Gifford or Briggs, time-varying wind) |
| Cross-sectional flux | $\int \mathrm{d}y$ on transects | steady state, full transects | wind speed and direction, shear | 50 m to 1 km | geotoolz `plume.CrossSectionalFlux` |
| Divergence | $\nabla_\text{h} \cdot \bar{\vect{F}}$ or $\oint$ | time average, no diffusion | gridded wind, many overpasses | 10 to 100 km | xrtoolz `budgets` (generic) |
| IME | $\iint \mathrm{d}A$, then $\tau_\text{res}$ | steady state, calibrated $U_\text{eff}$ | wind speed, plume mask, the calibration's $L$ | 50 m to 1 km | geotoolz `plume.IMEEstimate` |
:::

The divergence method and the cross-sectional flux ([](#tbl-er1-summary)) are the same statement in differential and in integral form. The IME is the area analogue of the cross-sectional flux, which trades the wind direction for a calibration. plumax also carries forward models that the note does not cover (a Lagrangian particle model and an Eulerian large-eddy model on finitevolx), which can generate the synthetic plumes an IME calibration is fitted on. The cross-sectional flux, the divergence method and the IME only use the column equation and not the Gaussian solution, so they hold for any plume shape.

(er1-open-questions)=
## Open questions

1. **Which length does the stack's IME calibration use?** geotoolz takes $U_\text{eff}$ as an input. The calibration and the length method should travel together as one object, so that a $\sqrt{A}$ coefficient cannot meet a chord length.
2. **Where should the divergence method live?** It is a budget over a methane column on a grid, so xrtoolz's budgets compute it, plumax owns the physics, and a TROPOMI pipeline owns the averaging.
3. **Can plumax calibrate $U_\text{eff}$ itself?** Its large-eddy model can generate the synthetic plumes which Varon et al. made with WRF, and a calibration workflow would make the coefficients reproducible per instrument.

(er1-todos)=
## Upcoming work

[](#tbl-er1-todos) lists the follow-ups.

:::{table} Follow-ups from checking the source note.
:label: tbl-er1-todos

| Follow-up | Where | Why |
|---|---|---|
| Add `length_method="sqrt_area"` | geotoolz `plume_length` (plumax after the move) | the note's and Varon et al.'s length is $\sqrt{A}$, which the operator cannot compute |
| Bundle $U_\text{eff}$ with its length method | `IMEEstimate` | stage 5: a factor of 3.7 between two lengths on one plume |
| Report $U_\text{eff}$ under shear in the cross-sectional flux | `CrossSectionalFlux` | stage 2: the effective wind is 15 % below the 10 m wind at 100 m and 21 % above it at 2 km |
| Add a methane divergence-method recipe on xrtoolz budgets | xrtoolz or plumax | nothing in the stack applies the method to columns |
:::

(er1-references)=
## References

- The source note, "Emission Rate Estimation from First Principles" [@geostack_plumax_note].
- The IME and its calibration [@varon2018], Sentinel-2 retrievals [@varon2021], the airborne flux distribution [@frankenberg2016], the cross-sectional flux [@krings2011], the divergence method [@beirle2019; @liu2021; @koene2024], the review [@jacob2022] and the plume and puff comparison [@jia2025].
- plumax, geotoolz, georeader and xrtoolz at the revisions in the header.
