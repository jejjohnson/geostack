---
title: "pyrox-ssm: state-space models as NumPyro models"
short_title: "pyrox-ssm"
description: "pyrox-ssm is the pyrox workspace's fifth package: temporal GPs and linear, nonlinear and ensemble state-space models as NumPyro models, whose inference is a recursion in time. Every piece of numerics it runs is gaussx's (SDE kernels, Kalman filters and smoothers, moment-matched nonlinear filters) or filterax's (ensemble assimilation); pyrox-ssm owns the sample sites, the factors and the choice of strategy. It sits in GeoML tier 3 with the rest of pyrox, and nothing in the stack imports it. Status: built, pyrox-ssm v0.2.0."
date: 2026-11-17
keywords:
  - pyrox-ssm
  - pyrox
  - state-space models
  - design
  - GeoML
---

pyrox-ssm is the pyrox workspace's fifth package: temporal GPs and linear, nonlinear and ensemble state-space models as NumPyro models, whose inference is a recursion in time. Every piece of numerics it runs is gaussx's (SDE kernels, Kalman filters and smoothers, moment-matched nonlinear filters) or filterax's (ensemble assimilation); pyrox-ssm owns the sample sites, the factors and the choice of strategy. It sits in GeoML tier 3 with the rest of pyrox, and nothing in the stack imports it. Status: built, pyrox-ssm v0.2.0.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | GeoML, tier 3; a package of the pyrox workspace (pyrox, -gp, -lgm, -nn, -ssm) |
| Imports | the pyrox core, gaussx (with its `numpyro` extra), NumPyro, optimistix, lineax, Equinox, einx, JAX; `[flows]` adds gauss-flows and FlowJAX, `[ensemble]` adds filterax |
| Imported by | nothing in the stack; pyrox-gp, pyrox-lgm and pyrox-nn never import it, and it never imports them |
| Owns | NumPyro priors, factors and sample sites over linear-Gaussian, Markov-GP, nonlinear and ensemble state-space models; the non-Gaussian Markov strategies; ragged-site and hierarchical helpers |
| Does not own | the numerics of any filter or smoother (gaussx); data-assimilation models and ensemble filters (filterax); dense and sparse GPs (pyrox-gp); GMRF components and `inla()` (pyrox-lgm) |
| Status | built; pyrox-ssm v0.2.0, released on 13 Nov 2026 |
| Tracked in | [geostack#144](https://github.com/jejjohnson/geostack/issues/144), with the pyrox epics [pyrox#356](https://github.com/jejjohnson/pyrox/issues/356) (the move out of pyrox-gp) and [pyrox#357](https://github.com/jejjohnson/pyrox/issues/357) (the capabilities for v1) |

The three model packages of pyrox split by what their inference does, not by the model they hold.

| Package | Inference | Holds |
| --- | --- | --- |
| pyrox-gp | linear algebra over all $N$ points at once | dense, sparse and SVGP models |
| pyrox-ssm | a recursion in time | Markov and linear-Gaussian SSM priors, factors, ELBOs, non-Gaussian Markov strategies |
| pyrox-lgm | a sparse Cholesky on one assembled precision | GMRF components, PC priors, `inla()` |

A stationary GP with a rational spectral density is an exact linear SDE, and a gap $\Delta t_k$ turns it into a linear-Gaussian chain with $A_k = \exp(F \Delta t_k)$ and $Q_k = P_\infty - A_k P_\infty A_k^\top$. Everything up to and including that chain, and the filter that runs on it, stays in gaussx. The same chain written as a block-tridiagonal precision is what pyrox-lgm's `MarkovGP` component uses: it calls `gaussx.ssm_to_naturals` directly and does not import pyrox-ssm.

## What the code holds

Read at pyrox `38feb4b`, gaussx `6f98b68` and filterax `807bb53`, the pins of geostack's submodules.

| Piece | Lives in | Names |
| --- | --- | --- |
| SDE kernels and their discretisation | gaussx | `MaternSDE`, `IntegratedWienerSDE`, `PeriodicSDE`, `QuasiPeriodicSDE`, `CosineSDE`, `ConstantSDE`, `SumSDE`, `ProductSDE`, `SDEKernel.discretise_sequence`, `sde_autocovariance` |
| Kalman filters and smoothers | gaussx | `kalman_filter`, `rts_smoother`, `parallel_kalman_filter`, `parallel_rts_smoother`, `infinite_horizon_filter`, `infinite_horizon_smoother`, `meanfield_kalman_filter`, `rts_interpolate` |
| Nonlinear filters and moment transforms | gaussx | `nonlinear_kalman_filter`, `nonlinear_rts_smoother`, `moment_match`, `moment_transform`, `UnscentedIntegrator`, `GaussHermiteIntegrator`, `CubatureIntegrator`, `TaylorIntegrator` |
| A chain as a precision | gaussx | `ssm_to_naturals`, `naturals_to_ssm`, `sites_to_precision` |
| Site updates for non-Gaussian likelihoods | gaussx | `site_natural_from_tilted`, `site_mean_var_from_natural`, `ep_tilted_moments`, `cvi_update_sites`, `newton_update` |
| Ensemble steps | gaussx | `enkf_analysis`, `etkf_transform`, `ensemble_kalman_gain`, `localized_kalman_gain` |
| Data-assimilation models and ensemble filters | filterax | `AbstractDynamics`, `filters.ETKF`, `filters.EnSRF`, `filters.ESTKF`, `filters.LETKF`, `differentiable_assimilate`, `EKI` |
| NumPyro shells | pyrox-ssm | `LinearGaussianSSMPrior`, `MarkovGPPrior`, `SparseMarkovGPPrior`, `SharedPlusSiteMarkovGP`, the `*_factor` and `*_sample` functions, the four Markov strategies |

pyrox-ssm re-exports gaussx's SDE kernels, so a Markov GP needs one import. Its top-level modules hold about 5.3k lines, and `pyrox_ssm.ensemble` and the `pyrox_ssm.flows` subpackage are the only parts that import an extra.

## NumPyro mechanics

Most of a state-space model goes into NumPyro with no help: the hyperparameters are sample sites, the Kalman log marginal is one `numpyro.factor`, and NUTS and SVI differentiate through the filter. `markov_gp_factor`, `lgssm_factor`, `sparse_markov_factor` and `nonlinear_ssm_factor` are each that one factor, with the conventions fixed in one place. Four pieces are not this simple, and pyrox-ssm owns them.

1. **Masks for ragged sites.** Every Markov entry point takes an observation mask, and a masked step runs the predict only, so it adds nothing to the log marginal. `pad_sites` pads per-site series to one `(S, N)` shape with strictly increasing padded times and returns that mask, and `markov_gp_sites_factor` scores the padded sites as one factor.
2. **Non-centred hierarchical scales.** `noncentred_log_normal` draws a positive per-site parameter as $\log \theta_n = m + \tau z_n$ with $z_n \sim \mathcal{N}(0, 1)$, which removes the funnel NUTS meets when the spread $\tau$ is small. `markov_gp_from_innovations` and `markov_gp_innovations_sample` do the same for a latent path: the path is run forward from whitened innovations in $O(N d^3)$, the non-centred form NUTS prefers.
3. **Jit-safe non-Gaussian fits.** `LaplaceMarkovInference`, `GaussNewtonMarkovInference`, `PosteriorLinearizationMarkov` and `ExpectationPropagationMarkov` run their site updates as `optimistix.fixed_point` with `optimistix.FixedPointIteration`, so a fit traces under `jax.jit`. `markov_implicit_adjoint` differentiates at the fixed point with a matrix-free GMRES, so a gradient does not grow with the number of iterations, and a failed solve returns a value instead of raising inside NUTS. `markov_laplace_factor` adds the approximate evidence as a factor.
4. **Fixed random numbers for ensemble likelihoods.** `ensemble_factor` scores the forecast ensemble of `filterax.differentiable_assimilate`. The initial ensemble is data, drawn once outside the model, and only deterministic square-root filters are accepted (filters and inflators that draw at each step raise), so the log density is a smooth function of $\theta$. The log likelihood is biased by the Monte Carlo moments, and the docstring asks for the ensemble size with any result.

## Examples

Each sketch is the model of one notebook in pyrox's `docs/notebooks/ssm/`, cut to its essential calls. The markers follow the house standard: `today (pkg)` names a call that ships at the pins above, and `user` a function the notebook defines.

### Example 1: one site

A Matérn-3/2 GP around a seasonal climatology $\mu(t)$ at one station, with masked days, fit by NUTS and SVI.

```python
# Shapes: N = 730 days; D_θ = 6 parameters

# today (pyrox_ssm)
# k_Matérn-3/2(t, t'; σ_f², ℓ) as a linear SDE
sde = MaternSDE(variance=variance, lengthscale=lengthscale, order=1)
# today (pyrox_ssm)
# μ + f ~ 𝒢𝒫(μ, k_Matérn-3/2) on the N days
prior = MarkovGPPrior(sde, times, mean_fn=lambda t: climatology(t, c, a, b))
# today (pyrox_ssm)
# log p(y ∣ θ) = ∑ₖ oₖ log 𝒩(yₖ; μ(tₖ) + H μ_x,k∣k−1, H Σ_x,k∣k−1 Hᵀ + σ_ε²)
markov_gp_factor("y", prior, y, noise_var, mask=mask)

# today (pyrox_ssm), per posterior draw, outside the model
# p(x_1:N ∣ y; θ) by the Kalman filter and RTS smoother
cond = prior.condition(y, noise_var, mask=mask)
# today (pyrox_ssm)
mean, var = cond.predict(t_star)                       # (M,) → (M,), (M,)
```

Notebook: `ssm_one_site.ipynb`.

### Example 2: hierarchical across sites

Six stations with records of different lengths share a seasonal cycle and a noise level, and each has its own kernel hyperparameters drawn around a population value.

```python
# Shapes: N_Ω = 6 stations; N = max_n N_n padded steps

# today (pyrox_ssm)
# T, Y, O with the padded steps masked, o_n,k = 1 for k ≤ N_n
times, y, mask = pad_sites(times_per_station, y_per_station)  # → 3 × (N_Ω, N)

with numpyro.plate("station", times.shape[0]):
    # today (pyrox_ssm)
    # σ_f,n² = exp(μ_θ,1 + τ_θ,1 z_n,1), z_n,1 ~ 𝒩(0, 1)
    variance = noncentred_log_normal("variance", variance_loc, variance_scale)
    # today (pyrox_ssm)
    # ℓ_n = exp(μ_θ,2 + τ_θ,2 z_n,2)
    lengthscale = noncentred_log_normal(
        "lengthscale", lengthscale_loc, lengthscale_scale
    )
    # user: MarkovGPPrior(...).log_marginal(y_n, σ_ε², mask=o_n) for station n
    per_station = jax.vmap(
        station_log_marginal, in_axes=(0, 0, 0, 0, 0, None, None, None)
    )
    # log p(y_n ∣ a, b, θ_n, σ_ε²), one Kalman filter per station
    log_marginals = per_station(                       # (N_Ω, N) → (N_Ω,)
        times, y, mask, variance, lengthscale, a, b, noise_var
    )
    # today (numpyro)
    # log p(Y ∣ ·) = ∑ₙ log p(y_n ∣ ·)
    numpyro.factor("y", log_marginals)
```

Notebook: `ssm_hierarchical_sites.ipynb`. The notebook writes `noncentred_log_normal` inline; `pyrox_ssm.noncentred_log_normal` has the same signature.

### Example 3: shared plus site

A regional signal $g$ shared by five stations plus a local anomaly $h_n$ at each, as one linear-Gaussian SSM on the merged time grid with state $x = (x_g, x_{h,1}, \dots, x_{h,N_Ω})$.

```python
# Shapes: N_Ω = 5 stations; T merged days; D_x = D_g + N_Ω D_h

# today (pyrox_ssm)
# 𝒯_y = ⋃ₙ 𝒯_y,n, Y with NaN where o_n,k = 0, and the (T, N_Ω) mask
network, y_grid = SharedPlusSiteMarkovGP.from_sites(  # → model, (T, N_Ω)
    regional_sde, local_sde, times_per_station, y_per_station
)

# inside the model, with θ sampled
# today (pyrox_ssm)
network = SharedPlusSiteMarkovGP(regional_sde, local_sde, times, mask)
# today (pyrox_ssm)
# log p(Y ∣ θ) = ∑ₖ log 𝒩(y_𝒪ₖ; H_𝒪ₖ μ_x,k∣k−1, H_𝒪ₖ Σ_x,k∣k−1 H_𝒪ₖᵀ + σ_ε² I)
log_marginal = network.log_marginal(y_grid, noise_var)  # (T, N_Ω) → ()
# today (numpyro)
numpyro.factor("y", log_marginal)

# today (pyrox_ssm)
# 𝔼[g(t_*) ∣ Y; θ], 𝔼[h_n(t_*) ∣ Y; θ]
g_mean, h_mean = network.decompose(y_dense, noise_var)  # → (M,), (M, N_Ω)
```

Notebook: `ssm_shared_plus_site.ipynb`.

### Example 3b: the full hierarchical model

Daily counts at eight facilities, with an intercept $c$, a spatial effect $g(s_n)$ from a Matérn GP over the facility coordinates, a temporal path $f_n$ per facility, and a Poisson likelihood. The spatial term is pyrox-gp's and the temporal term pyrox-ssm's: the user composes the two packages, which do not import each other.

```python
# Shapes: N_Ω = 8 facilities; T = 80 days; D_x = 2

# today (pyrox_gp)
# g ~ 𝒩(0, K_SS) at the facility coordinates S_y
spatial_prior = GPPrior(spatial_kernel, coords)
# today (pyrox_gp)
# g = L_SS z_g, z_g ~ 𝒩(0, I)
effect = gp_sample("facility_effect", spatial_prior, whitened=True)  # → (N_Ω,)
# today (pyrox_ssm)
# k_Matérn-3/2(t, t'; σ_f², ℓ) as a linear SDE, pooled over the facilities
temporal_sde = MaternSDE(temporal_variance, temporal_lengthscale, order=1)
# today (pyrox_ssm)
temporal_prior = MarkovGPPrior(temporal_sde, days)

with numpyro.plate("facility", coords.shape[0]):
    # today (pyrox_ssm)
    # f_n(t_k) = H x_n,k, x_n,k = A_k x_n,k−1 + L_Q,k ε_n,k
    path = markov_gp_innovations_sample("path", temporal_prior)  # → (N_Ω, T)
    # log ρ_n,k = c + g(s_n) + f_n(t_k)
    log_rate = einx.add("n, n k -> n k", intercept + effect, path)  # → (N_Ω, T)
    # ρ_n,k = exp(log ρ_n,k)
    rate = jnp.exp(log_rate)                           # → (N_Ω, T)
    # today (numpyro)
    # y_n,k ~ Poisson(ρ_n,k) where o_n,k = 1
    numpyro.sample("y", dist.Poisson(rate).mask(mask).to_event(1), obs=counts)
```

Notebook: `ssm_spatial_hierarchical.ipynb`, fit with NUTS and SVI.

### Example 4: non-Gaussian, collapsed and latent

Daily rain occurrence under a Bernoulli likelihood, two ways. Route A integrates the latent path out with a Laplace fit inside the model; Route B samples it from whitened innovations.

```python
# Shapes: N = 90 record days; T = 105 grid days; D_x = 2; D_θ = 3

# Route A: the path collapsed
# today (pyrox_gp)
# p(y_k ∣ f) = Bernoulli(σ(μ(t_k) + f(t_k)))
likelihood = BernoulliLikelihood()
# today (pyrox_ssm)
# damped fixed-point Newton on the sites, s ← (1 − α) s + α 𝒯(s), α = 0.5
laplace = LaplaceMarkovInference(damping=0.5, max_iter=50)
# today (pyrox_ssm)
# log p(y ∣ θ) ≈ log p(y ∣ f̂) + log 𝒩(f̂; μ, K_tt) − ½ log det(I + Λ_site K_tt)
markov_laplace_factor(
    "y", prior, likelihood, wet, strategy=laplace, mask=mask, reject_unconverged=True
)

# Route B: the path sampled
# today (pyrox_ssm)
# μ(t_k) + f(t_k) = μ(t_k) + H x_k, x_k = A_k x_k−1 + L_Q,k ε_k
logit = markov_gp_innovations_sample("logit", prior)  # → (T,)
# today (numpyro)
numpyro.sample("y", dist.Bernoulli(logits=logit).mask(mask).to_event(1), obs=wet)
```

Notebook: `ssm_nongauss.ipynb`, which cross-checks the two routes under NUTS. The likelihood is pyrox-gp's `BernoulliLikelihood`, which the notebook imports beside pyrox-ssm.

### Example 5: ensembles

5a scores an advection-diffusion model of a tracer on 64 cells from 8 sensors with the ensemble Kalman log likelihood. 5b fits the same model with an ensemble of variational fits from the pyrox core.

```python
# Shapes: N_Ex = 32 members; D_x = 64 cells; D_y = 8 sensors; T = 16 steps

# x₀⁽ᵉ⁾ ~ 𝒩(0, Σ_x,0), e ∈ ℰ_x, drawn once and passed in as data
white = jr.normal(key_ens, (N_MEMBERS, N_CELLS))       # → (N_Ex, D_x)
# user: colour maps white noise to a draw from 𝒩(0, Σ_x,0)
init_ensemble = jax.vmap(colour)(white)               # (N_Ex, D_x) → (N_Ex, D_x)

# 5a, inside the model
# user: a filterax.AbstractDynamics with the sampled u and K
dynamics = AdvectionDiffusion(speed, diffusivity)
# today (pyrox_ssm), calling filterax
# log p(y_1:T ∣ θ) ≈ ∑ₖ log 𝒩(r̄ₖ; 0, H̃ₖ H̃ₖᵀ + Σ_ε)
ensemble_factor(
    "y", flx.filters.ETKF(), dynamics, observe, init_ensemble, y, obs_times, obs_noise
)

# 5b
# today (pyrox)
# ν⁽ᵉ⁾⋆ = argmin_ν ℒ_ELBO(ν), one fit per e ∈ ℰ_ν, vmapped
fit_params, fit_losses = ensemble_vi(
    tracer_model, AutoNormal(tracer_model), ensemble_size=8, num_epochs=200,
    data=(init_ensemble, y), seed=jr.PRNGKey(7), learning_rate=0.05,
)
```

Notebook: `ssm_ensemble.ipynb`, which installs `pyrox-ssm[ensemble]` and compares the ensemble log likelihood with the exact Kalman one from `lgssm_factor` on the same linear model.

### Example 6: nonlinear SSM

The Lorenz-63 system, one RK4 step as the dynamics $f(x; \theta)$, observed in its first component, with the moment-matched log likelihood of the unscented filter.

```python
# Shapes: D_x = 3; D_y = 1; T = 120 steps; D_θ = 3

# today (gaussx)
UNSCENTED = gaussx.UnscentedIntegrator(alpha=1.0)
# today (pyrox_ssm), calling gaussx
# log p(y_1:T ∣ θ) ≈ ∑ₖ oₖ log 𝒩(yₖ; μ_y,k, Σ_y,k)
state = nonlinear_ssm_factor(
    "y", lambda x: lorenz_step(x, sigma, rho), observe,
    PROCESS_NOISE, noise_var * jnp.eye(1), y, INIT_MEAN, INIT_COV,
    integrator=UNSCENTED, mask=mask,
)
# today (gaussx), per posterior draw
# μ_x,k∣T, Σ_x,k∣T by the moment-matched RTS smoother
smoothed = gaussx.nonlinear_rts_smoother(state, dynamics, integrator=UNSCENTED)
```

Notebook: `ssm_nonlinear.ipynb`, which also runs `gaussx.TaylorIntegrator`, the extended filter.

## Decision record

The decisions of 10 Nov 2026 below are the ones carried by an issue, each in its body; the record lists only those.

| Decision | Issue |
| --- | --- |
| The Markov-GP track moves from pyrox-gp to pyrox-ssm as a hard break, with no deprecation window: pyrox-gp cannot re-export from a sibling, and a copy left behind would drift. | [pyrox#363](https://github.com/jejjohnson/pyrox/issues/363) |
| The numerics of the likelihood protocol go to gaussx, the protocol and its adapters to the pyrox core, and the two small checks are copied. | [pyrox#361](https://github.com/jejjohnson/pyrox/issues/361) |
| The protocol moves alone first, and the likelihood families are unified later in their own epic. | [pyrox#359](https://github.com/jejjohnson/pyrox/issues/359) |
| Inputs $X$ pass through `data` in `build(y, θ, data)`, as pyrox-lgm passes offsets, replacing `likelihood_accepts_inputs`. | [pyrox#375](https://github.com/jejjohnson/pyrox/issues/375) |
| gaussx gains a generic callable-density likelihood with no NumPyro import, and the NumPyro wrapping stays in pyrox. | [pyrox#376](https://github.com/jejjohnson/pyrox/issues/376), [gaussx#831](https://github.com/jejjohnson/gaussx/issues/831) |
| Observation masks are exposed now, as a first-class argument. | [pyrox#364](https://github.com/jejjohnson/pyrox/issues/364) |
| `LinearGaussianSSMPrior` over any $(A_k, H_k, Q_k, R_k)$ is in v1, with `MarkovGPPrior` as its special case. | [pyrox#365](https://github.com/jejjohnson/pyrox/issues/365) |
| The scalable latent path is a whitened innovations site, which is also the non-centred form NUTS prefers. | [pyrox#366](https://github.com/jejjohnson/pyrox/issues/366) |
| The Markov strategies run on optimistix fixed-point solvers with an implicit adjoint, which replaces the argminax route for these loops. | [pyrox#367](https://github.com/jejjohnson/pyrox/issues/367) |
| `nonlinear_ssm_factor` is in v1. | [pyrox#368](https://github.com/jejjohnson/pyrox/issues/368) |
| `ensemble_factor` is in v1 behind `[ensemble]`, fit with SVI or MAP by default, NUTS for small parameter sets, and filterax's EKI when the model cannot be differentiated. | [pyrox#369](https://github.com/jejjohnson/pyrox/issues/369) |
| Hierarchical models stay patterns, and pyrox-ssm ships helpers for the fiddly parts (`pad_sites`, `SharedPlusSiteMarkovGP`) rather than a hierarchical model object. | [pyrox#370](https://github.com/jejjohnson/pyrox/issues/370) |
| pyrox-lgm gets a precision-form `MarkovGP` component, built after a prototype reproduces `AR1` and the R-INLA `ar1` fixture; it calls gaussx directly and does not depend on pyrox-ssm. | [pyrox#358](https://github.com/jejjohnson/pyrox/issues/358) |
| pyrox may import filterax, and not the other way round, so pyrox moves to GeoML tier 3 beside vardax. | [geostack#147](https://github.com/jejjohnson/geostack/issues/147) |

## Sources

- pyrox at `38feb4b` (geostack's submodule pin): `packages/pyrox-ssm/pyproject.toml` and `src/pyrox_ssm/`, `packages/pyrox-lgm/src/pyrox_lgm/_components/_markov.py`, `packages/pyrox/src/pyrox/inference/_ensemble.py`, and the notebooks in `docs/notebooks/ssm/`.
- gaussx at `6f98b68`: `src/gaussx/__init__.py` and `src/gaussx/_ssm/`.
- filterax at `807bb53`: `src/filterax/__init__.py` and `src/filterax/filters/`.
- gauss_flows at `caa4be4`, for the `[flows]` extra only.
- The issue bodies of geostack#144, #146 and #147 and of pyrox#356 to #376.
