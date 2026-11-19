# Probability, distributions and stochastic processes

Densities, conditioning, expectations, distribution functions and extremes, and continuous-time processes.

## Probability

| Rule | Write | Shape of value |
| --- | --- | --- |
| true, data-generating density | $`p_\text{data}(\mathbf{x})`$ | $`\mathbb{R}_{>0}`$ |
| model density, fixed parameters after a semicolon | $`p(\mathbf{y} \mid \mathbf{x}; \boldsymbol{\theta})`$ | $`\mathbb{R}_{>0}`$ |
| model density, random parameters behind the bar | $`p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta})`$ | $`\mathbb{R}_{>0}`$ |
| approximate density | $`q(\mathbf{z} \mid \mathbf{x}; \boldsymbol{\nu})`$, not $`q_{\boldsymbol{\nu}}(\mathbf{z} \mid \mathbf{x})`$ | $`\mathbb{R}_{>0}`$ |
| sampling | $`\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_x, \boldsymbol{\Sigma}_x)`$ | — |
| density value | $`\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_x, \boldsymbol{\Sigma}_x)`$ | $`\mathbb{R}_{>0}`$ |
| probability of an event | $`\mathbb{P}(y > y_0)`$ | $`[0, 1]`$ |
| expectation | $`\mathbb{E}_{q(\mathbf{z})}[g(\mathbf{z})]`$ | shape of $`g`$ |
| variance; covariance of a vector | $`\mathbb{V}[x]`$; $`\mathbb{V}[\mathbf{x}]`$ | $`\mathbb{R}`$; $`\mathbb{R}^{D_x \times D_x}`$ |
| entropy | $`\mathbb{H}[p]`$ | $`\mathbb{R}`$ |
| mutual information | $`\mathbb{I}[\mathbf{x}; \mathbf{y}]`$ | $`\mathbb{R}_{\ge 0}`$ |
| KL divergence | $`\mathrm{KL}[q \Vert p]`$ | $`\mathbb{R}_{\ge 0}`$ |

- The semicolon means the same thing in densities as in functions, $`f(\mathbf{x}; \boldsymbol{\theta})`$ and $`\mathcal{J}(\mathbf{x}; \boldsymbol{\theta})`$: a fixed argument, not a random one.
- Conditioning uses `\mid`, never a bare `|`, which also splits GitHub table cells. KL uses `\Vert`, never `\|`.
- A density value is not a probability and can exceed 1: for $`y \sim \mathcal{N}(0, 0.1^2)`$, $`p(0) \approx 3.99`$ while $`\mathbb{P}(y > 0) = 0.5`$.
- The second argument of $`\mathcal{N}`$ is a variance or covariance, never a standard deviation.
- A random variable and its value share one symbol; capitals stay reserved for stacked samples.
- Operators on random variables are blackboard bold: $`\mathbb{E}`$, $`\mathbb{V}`$, $`\mathbb{P}`$, $`\mathbb{H}`$, $`\mathbb{I}`$. The only other blackboard letters are the number sets $`\mathbb{R}`$, $`\mathbb{N}`$, $`\mathbb{Z}`$, $`\mathbb{C}`$, so $`\mathbb{C}`$ is never covariance; a cross-covariance is the named $`\boldsymbol{\Sigma}_{xy}`$.
- Name the distribution of every expectation in its subscript. Prefer a named covariance ($`\boldsymbol{\Sigma}_x`$) over $`\mathbb{V}[\mathbf{x}]`$ once it exists.

```math
\mathcal{L}_\text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\nu}) = -\mathbb{E}_{q(\mathbf{z} \mid \mathbf{x}; \boldsymbol{\nu})}\left[\log p(\mathbf{x} \mid \mathbf{z}; \boldsymbol{\theta})\right] + \mathrm{KL}[q(\mathbf{z} \mid \mathbf{x}; \boldsymbol{\nu}) \Vert p(\mathbf{z})]
```

## Distributions and statistics

These rules hold for any distribution; extremes are one variant. Classic parameter letters are allowed under the per-page rules of `geoscience.md`, as named components of $`\boldsymbol{\theta}`$.

| Object, for scalar $`y`$ with parameters $`\boldsymbol{\theta}`$ | Write | Value |
| --- | --- | --- |
| density (pdf or pmf) | $`p(y; \boldsymbol{\theta})`$ | $`\mathbb{R}_{\ge 0}`$ |
| CDF | $`F(y; \boldsymbol{\theta}) = \mathbb{P}(y' \le y)`$ | $`[0, 1]`$ |
| survival (exceedance) | $`1 - F(y; \boldsymbol{\theta})`$ | $`[0, 1]`$ |
| quantile function at level $`\tau \in (0, 1)`$ | $`F^{-1}(\tau; \boldsymbol{\theta})`$ | $`\mathbb{R}`$ |
| empirical CDF, sample quantile | $`\hat{F}_N(y) = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}[y_n \le y]`$, $`\hat{F}_N^{-1}(\tau)`$ | $`[0, 1]`$, $`\mathbb{R}`$ |
| order statistics | $`y_{(1)} \le \ldots \le y_{(N)}`$ | $`\mathbb{R}`$ |
| marginals, copula | $`F_d(y_d)`$; $`F(\mathbf{y}) = C(F_1(y_1), \ldots, F_D(y_D))`$ | $`C: [0, 1]^D \to [0, 1]`$ |
| standard normal density, CDF | $`\mathcal{N}(y; 0, 1)`$, $`\Phi(y)`$ | $`\mathbb{R}_{>0}`$, $`[0, 1]`$ |
| significance level | $`\alpha`$ | $`(0, 1)`$ |

- $`y'`$ is a dummy draw from the same distribution. The CDF is italic with arguments, so it is never read as the matrix $`\mathbf{F}`$.
- Survival is never `\bar{F}`, which reads as a sample average; the standard normal density is never $`\phi`$, which is a basis function.
- Families: $`\mathcal{N}`$ and $`\mathcal{GP}`$ are calligraphic; every other family is upright, $`\mathrm{Gamma}(a, b)`$, $`\mathrm{Poisson}(\rho)`$, $`\mathrm{GEV}(\mu, \sigma, \xi)`$.

**Extremes.**

| Object | Write | In general terms |
| --- | --- | --- |
| block maximum of block $`b`$ | $`y_{\text{max},b} = \max_{t \in \mathcal{T}_y^{(b)}} y_t`$, blocks $`\mathcal{T}_y^{(b)}`$ (`indices.md`), count $`N_\text{block}`$ | a draw from $`F = \mathrm{GEV}`$ |
| GEV, GPD | $`\mathrm{GEV}(\mu, \sigma, \xi)`$, $`\mathrm{GPD}(\sigma_u, \xi)`$ | $`\boldsymbol{\theta} = (\mu, \sigma, \xi)`$ |
| shape | $`\xi`$ (Coles) | Hosking's $`\kappa = -\xi`$ has the opposite sign; quote it only when citing |
| threshold, exceedance rate | $`u`$, $`\rho_u`$ [yr⁻¹] | the intensity letter |
| return period, return level | $`R`$ [years], $`y_R = F^{-1}(1 - 1/R)`$ | a quantile at $`\tau = 1 - 1/R`$ |

```math
\hat{y}_R = F^{-1}(1 - 1/R; \hat{\boldsymbol{\theta}}) = \hat{\mu} + \frac{\hat{\sigma}}{\hat{\xi}} \left[ \left( -\log(1 - 1/R) \right)^{-\hat{\xi}} - 1 \right]
```

Here $`\hat{\boldsymbol{\theta}} = (\hat{\mu}, \hat{\sigma}, \hat{\xi})`$ are fitted GEV parameters, $`R`$ is in years and $`\hat{y}_R \in \mathbb{R}`$ has the units of $`y`$; the form holds for $`\hat{\xi} \neq 0`$.

## Stochastic processes

| Object | Write | Shape |
| --- | --- | --- |
| continuous-time process | $`\mathbf{x}(t)`$; a subscript is always a step, $`\mathbf{x}_k = \mathbf{x}(t_k)`$ | $`\mathbb{R}^{D_x}`$ |
| Brownian motion | $`\mathbf{b}(t)`$ | $`\mathbb{R}^{D_b}`$ |
| Itô SDE | $`\mathrm{d}\mathbf{x}(t) = f(\mathbf{x}(t), t) \mathrm{d}t + \mathbf{L}(\mathbf{x}(t), t) \mathrm{d}\mathbf{b}(t)`$ | $`\mathbf{L} \in \mathbb{R}^{D_x \times D_b}`$ |
| Brownian increment | $`\Delta \mathbf{b}_k = \mathbf{b}(t_k) - \mathbf{b}(t_{k-1}) \sim \mathcal{N}(\mathbf{0}, \Delta t_k \mathbf{I}_{D_b})`$ | $`\mathbb{R}^{D_b}`$ |
| Gaussian process | $`f \sim \mathcal{GP}(\mu, k)`$, kernel $`k(\mathbf{x}, \mathbf{x}')`$ | — |
| score, learned score | $`\nabla_{\mathbf{x}} \log p_t(\mathbf{x})`$, $`g_\text{score}(\mathbf{x}, t; \boldsymbol{\theta})`$ | $`\mathbb{R}^{D_x}`$ |
| point-process intensity | $`\rho(\mathbf{s}, t)`$ | $`\mathbb{R}_{\ge 0}`$ |

- The drift keeps the dynamics letter $`f`$. Stratonovich integrals are marked $`\circ \mathrm{d}\mathbf{b}(t)`$; Itô is the default.
- The kernel always has two arguments and the index $`k`$ never has any.
- $`p_t`$ is declared as the density at time $`t`$. A learned score is a map with a word label, never $`\mathbf{s}_\theta`$, which reads as a coordinate.
- The intensity is $`\rho`$ (Møller and Waagepetersen) because $`\lambda`$ is wavelength. On physics pages $`\rho`$ may also be air density (`geoscience.md`); a page uses one meaning.

### Linear SDEs and state-space GPs

A linear time-invariant SDE carries a $`\text{c}`$ subscript (continuous time), so the discretised transition keeps $`\mathbf{F}_t`$:

| Object | Write | Shape |
| --- | --- | --- |
| linear SDE | $`\mathrm{d}\mathbf{x}(t) = \mathbf{F}_\text{c} \mathbf{x}(t) \mathrm{d}t + \mathbf{L}_\text{c} \mathrm{d}\mathbf{b}(t)`$ | $`\mathbf{F}_\text{c} \in \mathbb{R}^{D_x \times D_x}`$, $`\mathbf{L}_\text{c} \in \mathbb{R}^{D_x \times D_b}`$ |
| white-noise spectral density | $`\mathbf{Q}_\text{c}`$ | $`D_b{\times}D_b`$ |
| diffusion covariance | $`\mathbf{B}_\text{c} = \mathbf{L}_\text{c} \mathbf{Q}_\text{c} \mathbf{L}_\text{c}^\top`$ | $`D_x{\times}D_x`$ |
| stationary covariance | $`\boldsymbol{\Sigma}_{x,\infty}`$, solving $`\mathbf{F}_\text{c} \boldsymbol{\Sigma}_{x,\infty} + \boldsymbol{\Sigma}_{x,\infty} \mathbf{F}_\text{c}^\top + \mathbf{B}_\text{c} = \mathbf{0}`$ | $`D_x{\times}D_x`$ |
| discretised transition, model noise | $`\mathbf{F}_t = \exp(\mathbf{F}_\text{c} \Delta t_t)`$, $`\boldsymbol{\Sigma}_{\eta,t} = \boldsymbol{\Sigma}_{x,\infty} - \mathbf{F}_t \boldsymbol{\Sigma}_{x,\infty} \mathbf{F}_t^\top`$ | $`D_x{\times}D_x`$ |
| linearised drift | $`f(\mathbf{x}, t) \approx \mathbf{F}_{\text{c},t} \mathbf{x} + \mathbf{c}_t`$ | $`\mathbf{c}_t \in \mathbb{R}^{D_x}`$ |

- A steady state takes $`\infty`$ in the time slot: $`\boldsymbol{\Sigma}_{x,\infty}`$, $`\mathbf{G}_\infty`$; its other intermediates take word labels ($`\boldsymbol{\Sigma}_{\text{pred},\infty}`$).
- The lag-one cross-covariance of a chain is the off-diagonal block $`[\boldsymbol{\Sigma}_x]_{t+1,t}`$, mirroring the precision blocks $`[\boldsymbol{\Lambda}_x]_{t,t}`$ and $`[\boldsymbol{\Lambda}_x]_{t+1,t}`$.
- Intermediates of a recursion with no role of their own take word labels in the subscript: $`\boldsymbol{\Sigma}_{\text{back},t}`$, $`\boldsymbol{\Sigma}_{\text{lin},t}`$; $`\boldsymbol{\Sigma}_{h,t} = \mathrm{Cov}[h(\mathbf{x}_t)]`$ names the function it belongs to.

### Information form, sites and natural gradients

| Object | Write | Notes |
| --- | --- | --- |
| Gaussian natural parameters | $`(\boldsymbol{\Lambda}_x \boldsymbol{\mu}_x, -\frac{1}{2} \boldsymbol{\Lambda}_x)`$ | never $`\boldsymbol{\eta}`$ (model noise) or $`\boldsymbol{\theta}`$ (parameters) |
| information vector with a possibly singular precision | $`\boldsymbol{\Lambda\mu}_\text{lik}`$, $`\boldsymbol{\Lambda\mu}_\text{site}`$, written as one symbol | the product form $`\boldsymbol{\Lambda}_x \boldsymbol{\mu}_x`$ only when $`\boldsymbol{\Lambda}_x`$ is invertible |
| EP site, cavity, tilted, marginal | $`\boldsymbol{\Lambda}_\text{site}`$; $`\mu_\text{cav}`$, $`\mu_\text{tilt}`$, $`\mu_\text{marg}`$ (and $`\sigma^2`$ alike) | |
| CVI variational parameters, step size | $`\boldsymbol{\nu}_t`$, $`\gamma`$ | $`\gamma`$ is a step size, as in `optimisation.md` |
