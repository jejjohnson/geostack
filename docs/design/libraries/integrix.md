---
title: "integrix: integration for JAX"
short_title: "integrix"
description: "integrix is the stack's integration library: one integral, I = ∫ f dμ, computed three ways (a rule, a scan or a march), with shared adaptivity and adjoints. It sits at Core tier 1 on functax's rules and gaussx's solves, extends diffrax with the stack's steppers, and takes over finitevolX's steppers and pipekit-jax's adjoints. Status: proposed, at design-sketch depth."
date: 2026-11-05
keywords:
  - integrix
  - design
  - GeoStack
---

integrix is the stack's integration library: one integral, I = ∫ f dμ, computed three ways (a rule, a scan or a march), with shared adaptivity and adjoints. It sits at Core tier 1 on functax's rules and gaussx's solves, extends diffrax with the stack's steppers, and takes over finitevolX's steppers and pipekit-jax's adjoints. Status: proposed, at design-sketch depth.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | Core, tier 1 |
| Imports | functax, gaussx; extends diffrax; JAX, Equinox, lineax, einx |
| Imported by | finitevolX, tracerax, radiomax, vardax, sesmax-core, plumax, radtrax |
| Takes over | finitevolX's 11 generic steppers (0.1.0 added `AB3Solver`) and their pure step functions; pipekit-jax's `truncated_scan` and `to_diffrax_adjoint`; generic step-size control from sesmax-core (copied from somax) |
| Draws on | quadax's adaptive rules, without a dependency |
| Status | proposed; no repository yet |
| Tracked in | the [layer map](../stack_layer_map.md); the companion [functax](functax.md) design |

```{figure} ../../images/libraries/integrix.svg
:alt: where integrix sits · Core tier 1
:width: 100%

Where integrix sits · core tier 1.
```

integrix sits one tier above the rules it sums with, so every model, numerics library and assimilation library that steps or integrates can import it. This page is the design; core types, algorithms and the code survey are in [integrix: design details](integrix_details.md).

## The problem

Every library in the stack integrates something, and each writes its own integrator. Turning one satellite methane plume into an emission rate takes six integrals; four are coded in different places today, and two are not written yet:

| Step in the retrieval | Integral | Over | Where it is coded today |
| --- | --- | --- | --- |
| Band radiance from a spectrum | L\_b = ∫ L(λ) R(λ) dλ | spectrum | plumax `radtran/srf.py`, radtrax `instrument/srf.py`, geoops `ApplySRF` |
| Optical depth down a column | τ(z) = ∫₀ᶻ k(z′) n(z′) dz′ | space, cumulative | not yet: radtrax's slab model has one layer; a layered column needs it |
| Radiance over directions | ∫ I(μ) μ dμ, Gauss in μ = cos θ | angle | not yet: radtrax's multi-stream solvers will need it |
| Integrated mass enhancement | IME = ∬ ΔΩ(x, y) dA | space | plumax `gauss_plume`, `gauss_puff` |
| Transport of the plume | dc/dt = F(c, t), stepped from t₀ to t₁ | time | finitevolX steppers, somax, plumax `les_fvm` |
| Uncertainty of the rate | 𝔼\[Q\] under the posterior | probability | plumax `population`: averages over NumPyro posterior draws |

The rate is then Q = IME · U / L, with U the wind speed and L the plume length. Each row is a sum of weights against samples, or a step that repeats one; each library has rebuilt the weights, the stepping or the adjoint on its own.

Three problems follow. The same hand-written trapezoid sums sit in plumax, radtrax and xrtoolz. finitevolX holds eleven generic stepper classes that have nothing to do with finite volumes, so a spectral or particle model that wants an SSP Runge–Kutta step has to import a finite-volume library. And the adjoints that make 4D-Var and plume inversions affordable (checkpointed and truncated backward sweeps) live in pipekit-jax, the pipeline library, rather than next to the steppers they differentiate.

## One integral, three operations

Every row of the table is an integral of a quantity f over a domain Ω with a measure μ:

```{math}
I = \int_\Omega f(x) \mathrm{d}\mu(x)
```

Here Ω is a time interval, a column, a band, a set of directions, an image or a probability space, and μ is length, area, a spectral response or a distribution. The problems differ, but they reduce to three operations:

| Operation | Computes | Linear in f? | Examples |
| --- | --- | --- | --- |
| Rule | I ≈ wᵀf, weights against samples | yes | band radiance, column amounts, integrated mass |
| Scan | Fₖ = Fₖ₋₁ + wₖ fₖ, a cumulative integral | yes, and parallel with `jax.lax.associative_scan` | optical depth, CDFs, accumulations |
| March | yₙ₊₁ = Φ(yₙ, tₙ, h), where the integrand depends on the solution | no: each step solves rather than sums | ODE and PDE time stepping |

A march is still a rule in time. A Butcher tableau's weights bᵢ are quadrature weights over one step, and Gauss–Legendre collocation is an implicit Runge–Kutta method. That is why quadrature procedures and time steppers belong in one library.

Two pieces of machinery are shared by all three:

- **Adaptivity.** Adaptive quadrature compares a Gauss–Kronrod pair; adaptive time stepping compares an embedded Runge–Kutta pair. Both estimate the error from two rules and refine where it is large.
- **Adjoints.** A rule's adjoint is its weights, a scan's is a reverse scan, and a march's is a backward march with checkpointing.

The weights w of a rule come from functax: they are a property of a basis on a domain. integrix owns what is done with them: summing, scanning, marching, adapting and differentiating.

## Modules and boundaries

Six modules: one per operation, plus adaptivity, adjoints and probability. integrix owns the machinery, not the domains: spectral responses stay in radtrax and plumax, mass enhancement in plumax and exceedance events in xtremax; they pass in weights, measures or integrands.

| Module | Owns | Comes from |
| --- | --- | --- |
| `integrix.rule` | I = wᵀf over any axis or set of axes, with a functax rule or caller's weights; batched over leading dims | new; ends the hand-written trapezoid sums in plumax, radtrax and xrtoolz |
| `integrix.scan` | cumulative integrals, forward and reverse, parallel by `associative_scan` | new |
| `integrix.march` | time steppers, eight as diffrax solvers and three (AB2, leapfrog, split-explicit) in a loop of their own: forward Euler, RK2 to RK4, SSP-RK, IMEX, Adams–Bashforth, leapfrog with Robert–Asselin filter, split-explicit; step-size control | finitevolX `timestepping/diffrax_solvers.py` (11 classes) and the pure step functions behind them; sesmax-core's `imex_stepsize_controller` (copied from somax) |
| `integrix.adapt` | adaptive quadrature: Gauss–Kronrod, tanh–sinh, Romberg, cumulative; error estimates shared with step control | new, on quadax's design without depending on it |
| `integrix.adjoint` | adjoints of scans and marches: reverse scans, checkpointing, truncated backpropagation through time | pipekit-jax `truncated_scan`, `to_diffrax_adjoint` |
| `integrix.expect` | expectations 𝔼\[f\] by Monte Carlo and quasi-Monte Carlo (Sobol, Halton), importance sampling | new |

### What integrix owns, and what it hands off

The test for every name: does it sum, scan, march or adapt over a domain, or differentiate one of those? Then integrix owns it.

| Neighbour | integrix owns | The neighbour owns | The line |
| --- | --- | --- | --- |
| functax | procedures that use rules | quadrature rules: nodes and weights of a basis on a domain | a rule is data; a procedure that sums, adapts or marches is integrix's |
| gaussx | expectations by sampling | Gaussian expectations: Gauss–Hermite, unscented, sigma points; the linear solves an implicit step needs | an integral under a Gaussian with a closed rule is gaussx's |
| diffrax | the stack's steppers, as diffrax solvers, and its adjoints | `diffeqsolve`, its solver and adjoint protocols | integrix extends diffrax, as gaussx extends lineax |
| argminax | adjoints of marches and scans | adjoints at an optimum: implicit differentiation, Gauss–Newton curvature | a backward sweep through time is integrix's; a derivative at a fixed point is argminax's |
| finitevolX, spectraldiffx | generic stepping | spatial discretisation, and semi-Lagrangian advection (field in, field out) | a scheme that steps any right-hand side is integrix's |
| pipekit-jax | the adjoint maths | composition: `JaxOperator`, `Rollout`, `DiffraxForwardModel` | pipekit composes steps; integrix computes them |
| the models | the integrator | integrands and measures: spectral responses, plume fields, turbulence | the model passes f and μ in |

### What moves in

| From | What | Its users then import |
| --- | --- | --- |
| finitevolX | 11 stepper classes: `ForwardEulerDfx`, `RK2Heun`, `RK3SSP`, `RK4Classic`, `SSP_RK2`, `SSP_RK104`, `IMEX_SSP2`, `AB2Solver`, `AB3Solver`, `LeapfrogRAFSolver`, `SplitExplicitRKSolver`, with their pure step functions; `SemiLagrangianSolver` and `solve_ocean_pde` stay | integrix: finitevolX, spectraldiffx, sesmax-core |
| pipekit-jax | `truncated_scan`, `to_diffrax_adjoint` | integrix; pipekit keeps composition |
| somax / sesmax-core | generic step-size control (`imex_stepsize_controller`); model terms stay | integrix |
| plumax, radtrax, xrtoolz | hand-written trapezoid sums | `integrix.rule` with functax rules |

## In use

The methane example end to end: a rule for band radiance and integrated mass, a scan for optical depth, and a march for transport with a checkpointed adjoint. Every name is proposed; the rules come from functax and the Runge–Kutta steppers subclass diffrax's.

```python
# Shapes: W wavelengths; Z layers; Ny × Nx plume pixels; K steps; C state size

import diffrax as dfx
import functax as fx
import integrix as ix

# --- rule: band radiance ---
# proposed (functax)
# wₗ for ∫ g(λ) dλ on the sampled wavelengths
band_rule = fx.quad.trapezoid(wavelength)  # (W,) → rule
# proposed (integrix)
# L_b = ∫ L(λ) R(λ) dλ / ∫ R(λ) dλ
band_radiance = ix.rule.weighted_mean(  # (…, W) → (…,)
    band_rule, radiance, weight=response
)

# --- scan: optical depth ---
# proposed (functax)
layer_rule = fx.quad.trapezoid(altitude)  # (Z,) → rule
# proposed (integrix)
# τ(z) = ∫₀ᶻ k(z′) n(z′) dz′
optical_depth = ix.scan.cumulative(layer_rule, absorption * density)  # (Z,) → (Z,)

# --- rule: integrated mass enhancement ---
# proposed (functax)
pixel_rule = fx.quad.cell_area(plume_grid)  # (Ny, Nx) → rule
# proposed (integrix)
# IME = ∬ ΔΩ(x, y) dA
mass = ix.rule.integrate(pixel_rule, enhancement)  # (Ny, Nx) → ()

# --- march: plume transport, differentiable ---
# not pinned (diffrax)
terms = dfx.ODETerm(advection_diffusion)
# proposed (integrix)
# yₙ₊₁ = Φ(yₙ, tₙ, h), three-stage SSP Runge–Kutta
stepper = ix.march.RK3SSP()
# proposed (integrix)
# backward sweep, storing every 50th state
adjoint = ix.adjoint.Checkpointed(every=50)
# not pinned (diffrax)
solution = dfx.diffeqsolve(  # (C,) → (K, C)
    terms,
    stepper,
    t0=0.0,
    t1=duration,
    dt0=step,
    y0=initial,
    adjoint=adjoint,
    saveat=dfx.SaveAt(ts=times),
)
```

The rules are weights from functax, so the same `ix.rule` call serves a band, a column and an image. The march is ordinary diffrax: integrix supplies the stepper and the adjoint, not a new solve loop.

## Integration across the stack

Seven libraries would import integrix, pipekit-jax would hand code to it, and spectraldiffx's examples could use it. Most changes are an import: the steppers keep their diffrax interface, and a rule replaces a hand-written sum.

| Consumer | Uses today | With integrix |
| --- | --- | --- |
| finitevolX | its own 11 diffrax solver classes | imports 11 of them from `integrix.march`; keeps `SemiLagrangianSolver`, which takes its interpolation from functax |
| spectraldiffx | nothing in the library; its example QG model (`scripts/qg_model.py`) steps with diffrax's `KenCarp4` | no library change; examples may use integrix's IMEX steppers |
| sesmax-core | generic stepping and `imex_stepsize_controller` inherited from somax | integrix steppers and step control; model terms stay |
| plumax | trapezoid sums for IME and SRFs; finite-volume stepping in `les_fvm`; NumPyro posterior averages | `ix.rule` with functax rules; integrix steppers; `ix.expect` |
| radtrax | SRF convolution by hand; a single-slab τ | `ix.rule` for bands; `ix.scan` for layered columns; angular rules when multi-stream lands |
| vardax | 4D-Var forward runs differentiated through pipekit-jax's helpers; its optimiser adjoints go to argminax | `integrix.adjoint` for the backward sweep through its forward model (decision 7) |
| tracerax, radiomax | planned | particle stepping (`ix.march`, SDE terms through diffrax); geometric integrals over a pixel |
| pipekit-jax | `truncated_scan`, `to_diffrax_adjoint` | hands them to integrix; keeps `Rollout` and `DiffraxForwardModel` |

The largest change is finitevolX's, and it is an import swap:

```python
# Today:
# today (finitevolX)
from finitevolx import RK3SSP

# After:
# proposed (integrix)
from integrix.march import RK3SSP
```

The class keeps its diffrax interface, so call sites do not change; finitevolX keeps a deprecated alias for one release.

## Decisions and open questions

Seven decided, two open. Each entry gives the position first, then the trade-off.

Decisions 10 to 13, on integrals over space, are in [integrix: integrals over space](integrix_spatial.md); all four were decided on 10 Nov 2026.

### 1. Name: integrix

**Decided (5 Nov 2026).** `integrix`, free on PyPI when checked. `integrax` is taken; `quadrax`, `marchax`, `cumulax` and `evolvax` were the other candidates.

### 2. Scope: all integration

**Decided (5 Nov 2026).** Time, space, spectrum, angle, exceedance and probability, not only quadrature and time stepping. All six reduce to the rule, scan and march above, and share adaptivity and adjoints.

### 3. Rules in functax, procedures here

**Decided (5 Nov 2026).** The nodes and weights of a rule are a property of a basis on a domain, so functax owns them. integrix owns what sums, scans, adapts or marches with them. Gaussian rules (Gauss–Hermite, unscented, sigma points) stay in gaussx.

### 4. Layer: Core tier 1

**Decided (5 Nov 2026, on the layer map).** integrix imports functax and gaussx, both Core tier 0, and extends diffrax. Every library that steps or integrates sits above it.

### 5. Adaptive quadrature: written here, on quadax's design

**Decided (5 Nov 2026).** Adaptive Gauss–Kronrod, tanh–sinh, Romberg and cumulative integrals are written in integrix, drawing on quadax's design, with no dependency on quadax. quadax's results serve as references in tests.

### 6. Semi-Lagrangian advection stays in finitevolX

**Decided (5 Nov 2026).** It discards its departure points after each step, so it is a field scheme, not a stepper. 11 of finitevolX's 12 stepper classes move (0.1.0 added AB3Solver); `SemiLagrangianSolver` stays and takes its interpolation from functax.

### 7. vardax's adjoints: argminax's, not integrix's

**Decided (5 Nov 2026), after reading vardax.** vardax's `KStepAdjoint` and `OneStepAdjoint` subclass `optimistix.AbstractAdjoint`: they differentiate through an inner minimiser (warm-up iterations under `stop_gradient`, then k live steps), not backward through time. They go to argminax, as the layer map says. integrix keeps the time adjoints of the forward run that 4D-Var's cost contains: checkpointed backward marches and truncated scans.

### 8. Steppers: extend diffrax or own a loop?

**Leaning: extend diffrax, with a small loop for multistep schemes.** Eight of the eleven steppers moving from finitevolX (AB3Solver among them) already subclass diffrax's solvers, so `diffeqsolve`, its step-size controllers and its adjoints keep working. The other three, `AB2Solver`, `LeapfrogRAFSolver` and `SplitExplicitRKSolver`, are plain `eqx.Module`s today, because multistep and split-explicit schemes carry state between steps that diffrax's solver protocol handles awkwardly. integrix keeps them as its own stepping loop, with the same adjoint options.

### 9. Monte Carlo here or in pyrox?

**Open.** Plain Monte Carlo and quasi-Monte Carlo expectations are integration and fit `integrix.expect`. Sampling from a posterior (MCMC, SVI) stays in NumPyro and pyrox; integrix only averages what they draw.

## Roadmap

Six waves, each a GitHub issue epic in the integrix repository once it exists. integrix needs functax's rules first, so wave 0 waits on functax's wave 0 (geometry) and its rules wave.

| Wave | Delivers | Ends | Needs |
| --- | --- | --- | --- |
| 0. Scaffold | the repository from the library template; `integrix.rule` and `integrix.scan` on functax rules; CI, capability index, `scipy.integrate` reference tests | — | functax rules |
| 1. Steppers move in | the 11 steppers from finitevolX, with deprecation aliases there; step-size control from sesmax-core | finitevolX's steppers; sesmax-core's generic step control | 0 |
| 2. Adjoints | `truncated_scan` and `to_diffrax_adjoint` from pipekit-jax; checkpointed backward marches; reverse scans | adjoint maths in pipekit-jax | 1 |
| 3. Adaptive quadrature | Gauss–Kronrod, tanh–sinh, Romberg and cumulative, on quadax's design, with quadax as a test reference | — | 0 |
| 4. Expectations | Monte Carlo, quasi-Monte Carlo (Sobol, Halton) and importance sampling | ad hoc averages in plumax | 0 |
| 5. Consumers switch | plumax, radtrax and xrtoolz sums onto `ix.rule`; sesmax-core and plumax `les_fvm` onto integrix steppers; vardax's forward runs onto `integrix.adjoint` | the hand-written sums and stepping in the stack | 1–4 |

Waves 1 and 3 can run side by side. Wave 5 is one issue per consumer, filed in that consumer's repository and linked to the integrix epic.

## Sources

- [Diffrax documentation](https://docs.kidger.site/diffrax/): the solver and adjoint protocols integrix extends.
- [quadax](https://github.com/f0uriest/quadax): adaptive quadrature in JAX, the design integrix's adaptive module draws on.
- [SciPy integrate reference](https://docs.scipy.org/doc/scipy/reference/integrate.html): the test references for rules and adaptive quadrature.
- [The GeoStack layer map](../stack_layer_map.md): integrix's layer, tier, imports and moves.
- functax: the rules integrix sums with.
- Code read on 5 Nov 2026 at geostack's submodule pins: finitevolX `_src/timestepping/diffrax_solvers.py` and `semi_lagrangian.py`; pipekit-jax `adjoints.py`; somax `_src/solvers.py`; plumax, radtrax and xrtoolz SRF and quadrature call sites; spectraldiffx `scripts/qg_model.py`.
