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

Hand-tuning viscosity, drag and wind amplitude until a run is stable and resolved is the wrong interface to an ocean model. This page builds somax models from dimensionless numbers instead, with resolution guards, and shows that non-dimensionalisation and standardisation are one per-field affine map shared by the model, the assimilation and the emulators. Same Gulf Stream QG example as the other pages.

+++

**Where it sits.** Infrastructure under every other page: the QG model of pages 2–6 and 8 is built this way, the 4D-Var control vector of pages 3, 4 and 9 is flattened through this map, and the learned components of pages 5, 6 and 8 see its O(1) output.

**Status.** Designed 2026-09-09, shipped in somax 2026-09-12 as ten merged PRs (#180–#189); see [Status](#nondim-status). Every `today` block on this page runs against somax `main` (`9cb19b3`).

## TL;DR: one affine map, three users

- **The right knobs are dimensionless.** A double gyre is set by its Rossby number, its planetary-vorticity gradient and the widths of its boundary layers, not by $\nu$, $\kappa$ and $\tau_0$ in SI units. `from_nondimensional` takes the first and derives the second.
- **Resolution becomes a check, not a trial.** Boundary-layer widths are given in units of $L$, so the factory compares them to $\Delta x$ before the run and refuses an unresolved Munk layer.
- **Non-dimensionalisation and standardisation are the same object:** a per-field affine map $x \mapsto (x - \ell)/s$ on the state pytree. Physics picks $(\ell, s)$ from scales; data picks them from sample moments. One class, `StateAffine`, serves both.
- **The same map fixes the assimilation.** Flattening $h \sim 10^3$ m next to $u \sim 10^{-1}$ m/s into one control vector gives a background covariance four orders of magnitude out of balance; flattening through the map gives O(1) entries.
- **Scale sets are canonical per equation family,** not universal: advective for QG, inertial for shallow water, planetary for the sphere, diffusive for pure diffusion.
- **paramax keeps parameters physical** (positive, bounded) while optimisers work unconstrained; **flowjax stays optional**, for learned priors only.

## 1. The problem

**Running example.** Barotropic QG over a Gulf Stream window: a basin of side $L = 10^6$ m at about 40° N, so $f_0 = 9.4\cdot10^{-5}$ s⁻¹ and $\beta = 1.75\cdot10^{-11}$ m⁻¹ s⁻¹, with eddy velocities $U = 0.5$ m s⁻¹, on a $64 \times 64$ grid ($\Delta x = 15.6$ km).

Built from SI coefficients, this model asks for a lateral viscosity $\nu$ (m² s⁻¹), a bottom drag $\kappa$ (s⁻¹) and a wind amplitude $\tau_0$. None of the three says what the user cares about: is the western boundary current resolved, is the run inertial or viscous, will it blow up? The usual loop is to guess $\nu$, run, watch the boundary current ring or smear, and guess again. Each new grid, basin or latitude restarts the loop.

The state has the same problem from the other side. A shallow-water state stacks $h$ (order $10^3$ m), $u$ and $v$ (order $10^{-1}$ m/s). Flattened as-is for an ensemble filter or a 4D-Var control vector, a single perturbation size $\sigma$ moves thickness and velocity by the same *absolute* amount, and a network trained on it spends its first layer learning the units.

## 2. Notation

- $L$, $U$, $H$, $T$: length, velocity, depth and time scales; $f_0$ the reference Coriolis parameter, $\beta$ its meridional gradient, $g$ gravity, $g'$ reduced gravity, $D$ a diffusivity, $a$ and $\Omega$ the planet radius and rotation rate.
- A prime marks a dimensionless quantity: $x' = x/L$, $t' = t/T$, $q' = q/(U/L)$.
- $q$ is the PV anomaly on T-points, $\psi$ the streamfunction with $\nabla^2\psi = q$, and $J(\cdot,\cdot)$ the Arakawa Jacobian (page 7).
- $\nu$ is the harmonic lateral viscosity, $\kappa$ the linear bottom drag, $\tau_0$ the wind-stress-curl amplitude.
- $Ro = U/(f_0 L)$ the Rossby number, $\hat\beta = \beta L^2/U$ the dimensionless planetary gradient, $Bu = g'H/(f_0 L)^2$ the Burger number, $Fr = U/\sqrt{gH}$ the Froude number, $\varepsilon = 4\Omega^2a^2/(gH)$ the Lamb parameter.
- $\delta_M = (\nu/\beta)^{1/3}/L$, $\delta_S = \kappa/(\beta L)$ and $\delta_I = (U/\beta)^{1/2}/L$ are the Munk, Stommel and inertial boundary-layer widths in units of $L$.
- $x$ is a state pytree, $\ell$ and $s$ pytrees of the same structure (the location and scale), and $\mathcal{A}_{\ell,s}(x) = (x - \ell)/s$, leaf by leaf, the affine map.

## 3. Non-dimensionalisation is a factory problem

Reading somax's right-hand sides showed that none of them knows about units: `vector_field` reads only grid spacings, constants, parameters and precomputed fields such as $\beta y$ and the wind curl. So a dimensionless model needs **no change to any dynamics**, only a second constructor that chooses $L = U = H = 1$ and fills the existing `create()` arguments from dimensionless numbers.

In the advective scale set, $T = L/U$ and $f_0 = 1/Ro$, and the running example's equation becomes

$$
\partial_{t'} q' + J(\psi', q' + \hat\beta y') = \delta_M^3\hat\beta \nabla'^2 q' - \delta_S\hat\beta \nabla'^2\psi' + \hat\tau \operatorname{curl}\tau'.
$$

| Input | Definition | Running example | Sets |
|---|---|---|---|
| `rossby` | $Ro = U/(f_0L)$ | $5.3\cdot10^{-3}$ | $f_0' = 1/Ro$ |
| `beta_hat` | $\hat\beta = \beta L^2/U$ | $35$ | $\beta'$ |
| `delta_M` | $(\nu/\beta)^{1/3}/L$ | $0.07$ (70 km) | $\nu' = \delta_M^3\hat\beta$; in SI $\nu \approx 6.0\cdot10^3$ m² s⁻¹ |
| `delta_S` | $\kappa/(\beta L)$ | $0.04$ (40 km) | $\kappa' = \delta_S\hat\beta$; in SI $\kappa = 7.0\cdot10^{-7}$ s⁻¹ |
| `delta_I` | $(U/\beta)^{1/2}/L$ | — (optional) | the wind amplitude, via Sverdrup balance |

**Guards.** Because $\delta_M$ and $\Delta x' = 1/64$ are now both lengths in units of $L$, the factory checks them before returning: `check_munk_width` fails below 2 cells and warns below 4; `check_stommel_width` fails below 1 and warns below 2. The running example has $\delta_M/\Delta x' = 4.5$ and $\delta_S/\Delta x' = 2.6$, so it builds silently; with $\delta_M = 0.05$ (3.2 cells) it builds with a warning. `check_resolution=False` opts out for deliberately coarse runs.

## 4. Scale sets: canonical per equation family

There is no single scale set that suits every model, and the design does not pretend there is. Each model family has one, recorded on `Scales.kind` so the transforms and the CLI can dispatch on it.

| Family | Constructor | $L$ | $T$ | $U$ | Numbers the factory takes |
|---|---|---|---|---|---|
| QG (barotropic, baroclinic, reparameterised) | `Scales.advective` | basin or eddy scale | $L/U$ | given | $Ro$, $\hat\beta$, $\delta_M$, $\delta_S$, $\delta_I$, per-mode $Bu$ |
| Shallow water (1-D, 2-D, multilayer) | `Scales.inertial` | given | $1/f_0$ | $f_0 L Ro$ | $Ro$, $Fr$, per-interface $Bu_k$, Ekman numbers |
| Spherical SWM and QG | `Scales.planetary` | $a$ | $1/\Omega$ | $2\Omega a Ro$ | $Ro$, one of $\varepsilon$ / $Bu$ / $Fr$, Ekman numbers |
| pde1d / pde2d (Burgers, Navier–Stokes, convection) | `Scales.advective` | given | $L/U$ | given | $Re$, $Pe$ |
| Pure diffusion | `Scales.diffusive` | given | $L^2/D$ | $D/L$ (bookkeeping only) | none: $D' = 1$ (added during implementation; there is no velocity to build $L/U$ from) |

Two things worth remembering about the sphere:

- $f(\varphi) = 2\Omega\sin\varphi$, so $2\Omega$ plays the role of $f_0$. Every rate coefficient picks up a factor 2: $\kappa' = 2 Ek$.
- The mid-latitude deformation radius $\sqrt{gH}/f_0$ diverges at the equator. The guard for global runs is the equatorial radius $L_{eq} = (c/\beta_{eq})^{1/2}$ with $c = \sqrt{gH}$ and $\beta_{eq} = 2\Omega/a$, i.e. $L_{eq}/a = Bu^{1/4}$, compared against the *widest* interior cell. There is no $\hat\beta$ on the sphere: $\beta = 2\Omega\cos\varphi/a$ is fixed by geometry.

## 5. Standardisation is the same map

Non-dimensionalising $q$ and standardising $q$ are both $\mathcal{A}_{\ell,s}$:

$$
q' = \mathcal{A}_{\ell,s}(q) = \frac{q - \ell}{s}, \text{ with }
\underbrace{(\ell, s) = (0, U/L)}_{\text{physics: from scales}}
\text{ or }
\underbrace{(\ell, s) = (\hat\mu_q, \hat\sigma_q)}_{\text{data: from samples}}.
$$

So somax ships one class, `StateAffine`, with `forward` ($\mathcal{A}$), `inverse` ($\mathcal{A}^{-1}$), `compose`, and two constructors, `from_scales` and `from_samples` (optionally per grid point, with a mask so land cells of zero variance get $s = \max(\hat\sigma, \epsilon)$). Leaves of $\ell$ and $s$ broadcast against the fields, so a multilayer state takes per-layer scales of shape `(nl, 1, 1)`.

**In the model.** `ScaledModel` integrates any model in transformed coordinates. For an affine map the chain rule is exact:

$$
\frac{dy}{dt'} = \frac{T}{s} f(t'T, s y + \ell), \text{ with } y = \mathcal{A}_{\ell,s}(x),
$$

so it serves both a dimensional model run in nondimensional units and a dimensional model run on standardised states (the ML-baseline case).

**In the assimilation.** `state_to_vector`, `make_ensemble` and the vardax and filterax bridges take an optional `transform`. Perturbing in transformed space turns a scalar $\sigma$ into the per-field covariance

$$
B = \sigma^2 \operatorname{diag}(s^2)
$$

in physical units, instead of $\sigma^2 I$ across fields four orders of magnitude apart.

**In the exports.** `apply_scale_metadata` writes the scale set and scalar $(\ell, s)$ to dataset attributes under `somax:scales:`, including $T$, which cannot be recovered from $L$ and $U$ because the families disagree about it. Re-dimensionalisation happens offline.

## 6. Parameters and densities: paramax and flowjax

Two libraries were evaluated for the parameter side.

- **paramax: adopted as a base dependency.** It is two small modules on top of equinox. `positive(x)` stores $\nu$, $\kappa$, $\tau_0$ in softplus space and `interval(x, lo, hi)` in logit space; `paramax.unwrap` reconstitutes them at right-hand-side time. Calibration, 4D-Var parameter estimation and hybrid training then descend in unconstrained space and can never produce a negative viscosity. Gradients are with respect to the unconstrained values; the `Params` docstring says so.
- **flowjax: an optional extra (`somax[flows]`).** Its `Affine` forces the scale through a trainable softplus, the opposite of what a fixed physical scale wants, and the log-determinant only matters when transforming *densities*. `StateAffine` therefore exposes flowjax-compatible `transform_and_log_det` / `inverse_and_log_det`, with $\log|\det| = -\sum\log s$ over leaves, and `somax.flows.learned_prior` composes a normalising flow onto a fixed physical scaling: the flow only ever sees O(1) numbers, and the prior is still over physical states.

## 7. Code sketch

The running example end to end: build the model from dimensionless numbers, take one step of $\Delta t' = 10^{-3}$ (2 000 s, since $T = L/U = 2\cdot10^6$ s), redimensionalise, then flatten and perturb for an ensemble. $\Phi_{\Delta t'}$ is the model's flow map, $\sigma$ the ensemble spread in transformed units, and $\varepsilon_e$ standard normal draws.

```python
import jax
import jax.numpy as jnp
from somax import Scales, StateAffine
from somax.da import make_ensemble, state_to_vector
from somax.models import BarotropicQG, BarotropicQGState

# Shapes: Ny = Nx = 66 (64 interior cells + ghost ring); E = 16 members

# today (somax)
# Ro = U/(f₀L), β̂ = βL²/U, δ_M = (ν/β)^{1/3}/L, δ_S = κ/(βL)
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
# xₑ = x + σ εₑ,  εₑ ~ 𝒩(0, I),  e = 1 … E
ensemble = make_ensemble(                          # → (E, Ny·Nx)
    q_si, jax.random.key(0), size=16, std=0.2, transform=nondim
)
```

What the block checks when run (it does, against `9cb19b3`): the flattened `x` equals $q'$ exactly, so the DA control vector is O(1) while the state handed back by `unravel` is in s⁻¹; `nondim.scale.q` is $U/L = 5\cdot10^{-7}$ s⁻¹; and the Munk and Stommel guards pass without a warning.

(nondim-status)=
## 8. Status

The plan below was written on 2026-09-09 and every PR landed on 2026-09-12.

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

User documentation: somax `content/notes/nondimensional_runs.md`; tutorial `content/tutorials/learned_prior_flowjax.py`.

## 9. Decisions

| Question | Decision |
|---|---|
| Re-dimensionalise diagnostics? | No. Diagnostics report in the model's units; `ConservationDriftMonitor` reports relative drift, which is scale-invariant. Revisit only for dimensional budgets offline. |
| One universal scale set? | No: canonical per equation family (§4). |
| Spherical models? | In scope; shipped (#188, #189). |
| Change `create()` signatures? | No. `from_nondimensional` composes them; nothing breaks. |
| flowjax as a core dependency? | No; optional `flows` extra. paramax is core. |

## 10. What to watch

- **Direction of the map.** `forward` is physical → transformed. Passing an already nondimensional state through `state_to_vector(…, transform)` silently scales it twice; redimensionalise with `inverse` first, as the sketch does.
- **Two examples are not a validation pair.** somax's dimensional `doublegyre_bt_qg` config and its nondimensional counterpart describe different basins ($\beta L/f_0 = 0.16$ against $\hat\beta Ro = 1$). A dimensional-versus-nondimensional regression must build both from the same numbers.
- **`unwrap` inside the RHS** rebuilds the parameter pytree every evaluation; XLA folds it, but a multilayer-SWM benchmark has not been recorded.
- **Per-grid-point standardisation** puts arrays the size of a field into $(\ell, s)$; they belong in the data, so the exporter writes attributes for scalar leaves only.

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

Nothing here belongs upstream of somax: scales and dimensionless numbers are properties of the model, not of the C-grid operators.
