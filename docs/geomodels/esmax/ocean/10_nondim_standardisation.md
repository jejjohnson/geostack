---
title: "Non-dimensional models and standardised states"
subtitle: "esmax · ocean · 10 · Build a model from dimensionless numbers, and scale its state with the same affine map that the assimilation and the emulators use"
short_title: "10 · Non-dimensional models"
description: "Hand-tuning viscosity, drag and wind amplitude until a run is stable and resolved is the wrong interface to an ocean model. This page builds somax models from dimensionless numbers instead, with resolution guards, and shows that non-dimensionalisation and standardisation are one per-field affine map shared by the model, the assimilation and the emulators. Same Gulf Stream QG example as the other pages; the plan has shipped in somax."
date: 2026-10-25
keywords:
  - non-dimensionalisation
  - standardisation
  - scaling
  - somax
  - paramax
  - flowjax
  - data assimilation
---

+++ {"part": "abstract"}

Hand-tuning viscosity, drag and wind amplitude until a run is stable and resolved is the wrong interface to an ocean model. In this page, we build somax models from dimensionless numbers instead, with resolution guards, and we show that non-dimensionalisation and standardisation are one per-field affine map which the model, the assimilation and the emulators share. We use the same Gulf Stream QG example as the other pages.

+++

**Where it sits.** Infrastructure under every other page: the QG model of pages 2 to 6 and 8 is built this way, the 4D-Var control vector of pages 3, 4 and 9 is flattened through this map, and the learned components of pages 5, 6 and 8 see its O(1) output.

**Status.** We designed this on 2026-09-09 and it shipped in somax on 2026-09-12 as ten merged PRs (#180 to #189); see [Status](#nondim-status). Every `today` block on this page runs against somax `main` (`9cb19b3`).

## TL;DR: one affine map, three users

- **The right knobs are dimensionless.** A double gyre is set by its Rossby number, its planetary-vorticity gradient and the widths of its boundary layers, not by $\nu$, $\kappa$ and $\tau_0$ in SI units. `from_nondimensional` takes the first and derives the second.
- **Resolution becomes a check, not a trial.** The boundary-layer widths are given in units of $L$, so the factory compares them to $\Delta s$ before the run and refuses an unresolved Munk layer.
- **Non-dimensionalisation and standardisation are the same object**, which is a per-field affine map $x \mapsto (x - \ell)/\sigma$ on the state pytree. Physics picks $(\ell, \sigma)$ from scales, whereas data picks them from sample moments. So one class, `StateAffine`, serves both.
- **The same map fixes the assimilation.** Flattening $h \sim 10^3$ m next to $u \sim 10^{-1}$ m/s into one control vector gives a background covariance which is four orders of magnitude out of balance, whereas flattening through the map gives O(1) entries.
- **Scale sets are canonical per equation family,** not universal: advective for QG, inertial for shallow water, planetary for the sphere, and diffusive for pure diffusion.
- **paramax keeps parameters physical** (positive, bounded) while the optimisers work unconstrained, and **flowjax stays optional**, for learned priors only.

## 1. The problem

**Running example.** We take barotropic QG over a Gulf Stream window: a basin of side $L = 10^6$ m at about 40° N, so $f_0 = 9.4 \cdot 10^{-5}$ s⁻¹ and $\beta = 1.75 \cdot 10^{-11}$ m⁻¹ s⁻¹, with eddy velocities $U = 0.5$ m s⁻¹, on a $64 \times 64$ grid ($\Delta s = 15.6$ km).

Built from SI coefficients, this model asks for a lateral viscosity $\nu$ (m² s⁻¹), a bottom drag $\kappa$ (s⁻¹) and a wind amplitude $\tau_0$. However, none of the three says what the user cares about: is the western boundary current resolved, is the run inertial or viscous, and will it blow up? The usual loop is to guess $\nu$, run, watch the boundary current ring or smear, and guess again, and each new grid, basin or latitude restarts the loop.

The state has the same problem from the other side. A shallow-water state stacks $h$ (order $10^3$ m) with $u$ and $v$ (order $10^{-1}$ m/s). Flattened as-is for an ensemble filter or a 4D-Var control vector, a single perturbation size $\alpha$ moves thickness and velocity by the same *absolute* amount, and a network trained on it spends its first layer learning the units.

## 2. Notation

Every symbol below is a scalar unless the line says otherwise.

- $L$, $U$, $H$ and $T$ are the length, velocity, depth and time scales. $f_0$ is the reference Coriolis parameter, $\beta$ its meridional gradient, $g$ gravity, $g_\text{r}$ reduced gravity (often written $g'$), $D$ a diffusivity, and $a$ and $\Omega$ the planet radius and rotation rate.
- $\vect{s} = (s_1, s_2) \in \mathbb{R}^2$ are the zonal and meridional coordinates (m) and $\Delta s$ the grid spacing.
- A prime marks a dimensionless quantity: $\vect{s}' = \vect{s}/L$, $t' = t/T$, $q' = q/(U/L)$, and $\nabla' := \nabla_{\vect{s}'}$.
- $q(\vect{s}, t)$ is the PV anomaly on T-points, $\psi(\vect{s}, t)$ the streamfunction with $\nabla^2_{\vect{s}} \psi = q$, and $J(\cdot, \cdot)$ the Arakawa Jacobian (page 7).
- $\nu$ is the harmonic lateral viscosity, $\kappa$ the linear bottom drag and $\tau_0$ the wind-stress-curl amplitude.
- $\mathrm{Ro} = U/(f_0 L)$ is the Rossby number, $\beta' = \beta L^2/U$ the dimensionless planetary gradient, $\mathrm{Bu} = g_\text{r} H/(f_0 L)^2$ the Burger number, $\mathrm{Fr} = U/\sqrt{g H}$ the Froude number and $\varepsilon = 4 \Omega^2 a^2/(g H)$ the Lamb parameter.
- $\delta_\text{M} = (\nu/\beta)^{1/3}/L$, $\delta_\text{S} = \kappa/(\beta L)$ and $\delta_\text{I} = (U/\beta)^{1/2}/L$ are the Munk, Stommel and inertial boundary-layer widths in units of $L$.
- $x$ is a state pytree, and $\ell$ and $\sigma$ are pytrees of the same structure (the location and the scale). The affine map is $\mathcal{A}_{\ell,\sigma}[x] = (x - \ell)/\sigma$, leaf by leaf.

## 3. Non-dimensionalisation is a factory problem

Reading somax's right-hand sides showed that none of them knows about units: `vector_field` only reads grid spacings, constants, parameters and precomputed fields such as $\beta s_2$ and the wind curl. So a dimensionless model needs **no change to any dynamics**, only a second constructor which chooses $L = U = H = 1$ and fills the existing `create()` arguments from dimensionless numbers.

In the advective scale set, $T = L/U$ and $f_0' = 1/\mathrm{Ro}$, and the running example's equation becomes

$$
\partial_{t'} q' + J(\psi', q' + \beta' s_2') = \delta_\text{M}^3 \beta' \nabla'^2 q' - \delta_\text{S} \beta' \nabla'^2 \psi' + \tau_0' \mathrm{curl} \tau' ,
$$ (nondim-eq-qg)

where $\tau_0'$ is the dimensionless wind amplitude and $\tau'$ the dimensionless wind-stress pattern. [CHECK: the source wrote this coefficient as $\hat{\tau}$ without defining it; we read it as the dimensionless wind amplitude, and wrote $f_0' = 1/\mathrm{Ro}$ where the source had $f_0 = 1/Ro$, following the table below.]

| Input | Definition | Running example | Sets |
|---|---|---|---|
| `rossby` | $\mathrm{Ro} = U/(f_0 L)$ | $5.3 \cdot 10^{-3}$ | $f_0' = 1/\mathrm{Ro}$ |
| `beta_hat` | $\beta' = \beta L^2/U$ | $35$ | $\beta'$ |
| `delta_M` | $(\nu/\beta)^{1/3}/L$ | $0.07$ (70 km) | $\nu' = \delta_\text{M}^3 \beta'$; in SI $\nu \approx 6.0 \cdot 10^3$ m² s⁻¹ |
| `delta_S` | $\kappa/(\beta L)$ | $0.04$ (40 km) | $\kappa' = \delta_\text{S} \beta'$; in SI $\kappa = 7.0 \cdot 10^{-7}$ s⁻¹ |
| `delta_I` | $(U/\beta)^{1/2}/L$ | — (optional) | the wind amplitude, via Sverdrup balance |

**Guards.** Because $\delta_\text{M}$ and $\Delta s' = 1/64$ are now both lengths in units of $L$, the factory checks them before returning: `check_munk_width` fails below 2 cells and warns below 4, and `check_stommel_width` fails below 1 and warns below 2. The running example has $\delta_\text{M}/\Delta s' = 4.5$ and $\delta_\text{S}/\Delta s' = 2.6$, so it builds silently, whereas with $\delta_\text{M} = 0.05$ (3.2 cells) it builds with a warning. `check_resolution=False` opts out for deliberately coarse runs.

## 4. Scale sets: canonical per equation family

There is no single scale set that suits every model, and the design does not pretend there is. Each model family has one, which is recorded on `Scales.kind` so that the transforms and the CLI can dispatch on it.

| Family | Constructor | $L$ | $T$ | $U$ | Numbers the factory takes |
|---|---|---|---|---|---|
| QG (barotropic, baroclinic, reparameterised) | `Scales.advective` | basin or eddy scale | $L/U$ | given | $\mathrm{Ro}$, $\beta'$, $\delta_\text{M}$, $\delta_\text{S}$, $\delta_\text{I}$, per-mode $\mathrm{Bu}$ |
| Shallow water (1-D, 2-D, multilayer) | `Scales.inertial` | given | $1/f_0$ | $f_0 L \mathrm{Ro}$ | $\mathrm{Ro}$, $\mathrm{Fr}$, per-interface $\mathrm{Bu}_k$, Ekman numbers |
| Spherical SWM and QG | `Scales.planetary` | $a$ | $1/\Omega$ | $2 \Omega a \mathrm{Ro}$ | $\mathrm{Ro}$, one of $\varepsilon$ / $\mathrm{Bu}$ / $\mathrm{Fr}$, Ekman numbers |
| pde1d / pde2d (Burgers, Navier–Stokes, convection) | `Scales.advective` | given | $L/U$ | given | $\mathrm{Re}$, $\mathrm{Pe}$ |
| Pure diffusion | `Scales.diffusive` | given | $L^2/D$ | $D/L$ (bookkeeping only) | none: $D' = 1$ (added during implementation; there is no velocity to build $L/U$ from) |

There are two things worth remembering about the sphere:

- $f(\phi) = 2 \Omega \sin \phi$ at latitude $\phi$, so $2 \Omega$ plays the role of $f_0$. Every rate coefficient then picks up a factor of 2, as in $\kappa' = 2 \mathrm{Ek}$ with $\mathrm{Ek}$ the Ekman number.
- The mid-latitude deformation radius $\sqrt{g H}/f_0$ diverges at the equator. So the guard for global runs is the equatorial radius $L_\text{eq} = (c/\beta_\text{eq})^{1/2}$ with $c = \sqrt{g H}$ and $\beta_\text{eq} = 2 \Omega/a$ (i.e. $L_\text{eq}/a = \mathrm{Bu}^{1/4}$), which is compared against the *widest* interior cell. There is no $\beta'$ on the sphere, as $\beta = 2 \Omega \cos \phi / a$ is fixed by geometry.

## 5. Standardisation is the same map

Non-dimensionalising $q$ and standardising $q$ are both $\mathcal{A}_{\ell,\sigma}$:

$$
q' = \mathcal{A}_{\ell,\sigma}[q] = \frac{q - \ell}{\sigma} ,
$$ (nondim-eq-affine)

where physics takes $(\ell, \sigma) = (0, U/L)$ from the scales and data takes $(\ell, \sigma) = (\hat{\mu}_q, \hat{\sigma}_q)$ from the sample mean and standard deviation.

So somax ships one class, `StateAffine`, with `forward` ($\mathcal{A}$), `inverse` ($\mathcal{A}^{-1}$), `compose`, and two constructors, `from_scales` and `from_samples`. The latter can work per grid point, with a mask, so that land cells of zero variance get $\sigma = \max(\hat{\sigma}, \mathrm{tol})$. The leaves of $\ell$ and $\sigma$ broadcast against the fields, so a multilayer state takes per-layer scales of shape `(nl, 1, 1)`.

**In the model.** `ScaledModel` integrates any model in transformed coordinates. For an affine map the chain rule is exact:

$$
\frac{\mathrm{d}x'}{\mathrm{d}t'} = \frac{T}{\sigma} \mathcal{F}[\sigma x' + \ell](t' T) ,
$$ (nondim-eq-scaled)

where $x' = \mathcal{A}_{\ell,\sigma}[x]$ and $\mathcal{F}$ is the model's right-hand side, $\partial_t x = \mathcal{F}[x](t)$. So it serves both a dimensional model run in nondimensional units and a dimensional model run on standardised states (the ML-baseline case).

**In the assimilation.** `state_to_vector`, `make_ensemble` and the vardax and filterax bridges take an optional `transform`. Perturbing in the transformed space turns a scalar spread $\alpha$ into the per-field covariance

$$
\gmat{\Sigma}_x = \alpha^2 \mathrm{diag}(\gvect{\sigma}^2)
$$ (nondim-eq-cov)

in physical units, where $\gvect{\sigma} \in \mathbb{R}^{N_y N_x}$ is the scale pytree flattened like the state and $\gmat{\Sigma}_x \in \mathbb{R}^{N_y N_x \times N_y N_x}$. This replaces $\alpha^2 \mat{I}$ across fields which are four orders of magnitude apart.

**In the exports.** `apply_scale_metadata` writes the scale set and the scalar $(\ell, \sigma)$ to dataset attributes under `somax:scales:`, including $T$, which cannot be recovered from $L$ and $U$ because the families disagree about it. Re-dimensionalisation happens offline.

## 6. Parameters and densities: paramax and flowjax

We evaluated two libraries for the parameter side.

- **paramax: adopted as a base dependency.** It is two small modules on top of equinox. `positive(x)` stores $\nu$, $\kappa$ and $\tau_0$ in softplus space and `interval(x, lo, hi)` stores them in logit space, and `paramax.unwrap` reconstitutes them at right-hand-side time. So calibration, 4D-Var parameter estimation and hybrid training all descend in unconstrained space and can never produce a negative viscosity. The gradients are with respect to the unconstrained values, and the `Params` docstring says so.
- **flowjax: an optional extra (`somax[flows]`).** Its `Affine` forces the scale through a trainable softplus, which is the opposite of what a fixed physical scale wants, and the log-determinant only matters when transforming *densities*. `StateAffine` therefore exposes the flowjax-compatible `transform_and_log_det` / `inverse_and_log_det`, with $\log \vert \det \vert = -\sum \log \sigma$ over the leaves, and `somax.flows.learned_prior` composes a normalising flow onto a fixed physical scaling. So the flow only ever sees O(1) numbers, and the prior is still over physical states.

## 7. Code sketch

This is the running example end to end: we build the model from dimensionless numbers, take one step of $\Delta t' = 10^{-3}$ (2 000 s, since $T = L/U = 2 \cdot 10^6$ s), redimensionalise, and then flatten and perturb for an ensemble. Here $\Phi_{\Delta t'}$ is the model's flow map, $\alpha$ the ensemble spread in transformed units, and $\gvect{\xi}^{(e)} \in \mathbb{R}^{N_y N_x}$ the standard normal draws for member $e$.

```python
import jax
import jax.numpy as jnp
from somax import Scales, StateAffine
from somax.da import make_ensemble, state_to_vector
from somax.models import BarotropicQG, BarotropicQGState

# Shapes: Ny = Nx = 66 (64 interior cells + ghost ring); E = 16 members

# today (somax)
# Ro = U/(f₀L), β′ = βL²/U, δ_M = (ν/β)^{1/3}/L, δ_S = κ/(βL)
model, unit_scales = BarotropicQG.from_nondimensional(
    nx=64, ny=64, rossby=5.3e-3, beta_hat=35.0, delta_M=0.07, delta_S=0.04,
)

# today (somax)
# L = 10⁶ m, U = 0.5 m s⁻¹, f₀ = 9.4·10⁻⁵ s⁻¹, T = L/U
gulf_stream = Scales.advective(L=1.0e6, U=0.5, f0=9.4e-5)

# today (somax)
# forward: q′ = q/(U/L);  inverse: q = (U/L) q′
nondim = StateAffine.from_scales(BarotropicQGState, gulf_stream)

# user
# q′(t′ = 0) = 0
q_rest = BarotropicQGState(q=jnp.zeros((66, 66)))  # → (Ny, Nx)

# today (somax)
# q′(Δt′) = Φ_Δt′(q′(0)),  Δt′ = Δt/T
q_next = model.step(q_rest, 1.0e-3)                # (Ny, Nx) → (Ny, Nx)

# today (somax)
# q = (U/L) q′
q_si = nondim.inverse(q_next)                      # (Ny, Nx) → (Ny, Nx)

# today (somax)
# x = vec(q/(U/L))
x, unravel = state_to_vector(q_si, nondim)         # (Ny, Nx) → (Ny·Nx,)

# today (somax)
# x⁽ᵉ⁾ = x + α ξ⁽ᵉ⁾,  ξ⁽ᵉ⁾ ~ 𝒩(0, I),  e = 1, …, E
ensemble = make_ensemble(                          # → (E, Ny·Nx)
    q_si, jax.random.key(0), size=16, std=0.2, transform=nondim
)
```


When we run the block (against `9cb19b3`), it checks three things: 1) the flattened `x` equals $q'$ exactly, so the DA control vector is O(1) while the state handed back by `unravel` is in s⁻¹, 2) `nondim.scale.q` is $U/L = 5 \cdot 10^{-7}$ s⁻¹, and 3) the Munk and Stommel guards pass without a warning.

(nondim-status)=
## 8. Status

We wrote the plan below on 2026-09-09, and every PR landed on 2026-09-12.

| PR | somax | Change | Deviation from the plan |
|---|---|---|---|
| 1 | #180 | `Scales`, `StateAffine` (`from_scales`, `from_samples`, `compose`) | `Scales.diffusive` added; `Scales.planetary` shipped early, as planned |
| 2 | #181 | paramax: `positive`, `interval`, `unwrap` at RHS time, `Params` docstring | `NonTrainable` wrappers mask arrays only; interval span is bounded |
| 3 | #182 | `BarotropicQG.from_nondimensional` + Munk / Stommel guards | Guards live in `somax/_src/core/resolution.py`, not the CLI assertions; factories return `(model, Scales)` |
| 4 | #183 | `ScaledModel` (`from_scales`, `standardised`) | — |
| 5 | #184 | `from_nondimensional` for nonlinear SWM, multilayer SWM, baroclinic and reparameterised QG | — |
| 6 | #185 | `from_nondimensional` for pde1d / pde2d | — |
| 7 | #186 | `transform=` through DA flatten, ensembles and both bridges; `apply_scale_metadata`; CLI `nondim:` block | CLI rejects a config with both `consts:` and `nondim:`, and rejects dimensional knobs that the numbers derive |
| 8 | #187 | `somax[flows]`: `StateAffineBijection`, `learned_prior` | — |
| 9 | #188 | Spherical SWM and barotropic QG models, on finitevolX's spherical operators | Not in the original plan's scope for somax: the plan assumed the spherical model classes would arrive separately (epic #72 D4); they were built here instead |
| 10 | #189 | Spherical `from_nondimensional` + equatorial deformation-radius guard | Stratification given as exactly one of `lamb`, `burger`, `froude` |

The user documentation is somax's `content/notes/nondimensional_runs.md`, and the tutorial is `content/tutorials/learned_prior_flowjax.py`.

## 9. Decisions

| Question | Decision |
|---|---|
| Re-dimensionalise diagnostics? | No. Diagnostics report in the model's units; `ConservationDriftMonitor` reports relative drift, which is scale-invariant. Revisit only for dimensional budgets offline. |
| One universal scale set? | No: canonical per equation family (§4). |
| Spherical models? | In scope; shipped (#188, #189). |
| Change `create()` signatures? | No. `from_nondimensional` composes them; nothing breaks. |
| flowjax as a core dependency? | No; optional `flows` extra. paramax is core. |

## 10. What to watch

- **Direction of the map.** `forward` is physical → transformed. Passing an already nondimensional state through `state_to_vector(…, transform)` silently scales it twice, so redimensionalise with `inverse` first, as the sketch does.
- **Two examples are not a validation pair.** somax's dimensional `doublegyre_bt_qg` config and its nondimensional counterpart describe different basins ($\beta L/f_0 = 0.16$ against $\beta' \mathrm{Ro} = 1$), so a dimensional-versus-nondimensional regression must build both from the same numbers.
- **`unwrap` inside the RHS** rebuilds the parameter pytree at every evaluation. XLA folds it, but a multilayer-SWM benchmark has not been recorded.
- **Per-grid-point standardisation** puts arrays the size of a field into $(\ell, \sigma)$. They belong in the data, so the exporter only writes attributes for scalar leaves.

## 11. Where it lives

| Piece | Library | Module |
|---|---|---|
| `Scales`, `StateAffine` | somax | `somax/_src/core/scales.py`, `somax/_src/core/transforms.py` |
| `ScaledModel` | somax | `somax/_src/core/scaled.py` |
| Resolution guards | somax | `somax/_src/core/resolution.py` (equatorial guard in `somax/_src/cli/_assertions.py`) |
| `from_nondimensional` | somax | each model module under `somax/_src/models/` |
| `positive`, `interval` | somax on paramax | `somax/_src/core/types.py` |
| DA transforms | somax | `somax/_src/da/flatten.py`, `vardax_bridge.py`, `filterax_bridge.py` |
| Learned priors | somax on flowjax (optional) | `somax/flows.py` |

Nothing here belongs upstream of somax, as scales and dimensionless numbers are properties of the model and not of the C-grid operators.
