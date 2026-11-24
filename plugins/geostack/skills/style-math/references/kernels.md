# Kernels, sparse and variational GPs

## Kernels

**Vocabulary.**

| Word | Meaning |
| --- | --- |
| kernel | the function $`k(\mathbf{x}, \mathbf{x}')`$, always with two arguments |
| kernel matrix (Gram matrix) | $`k`$ evaluated on stacked inputs, $`\mathbf{K}_{XX}`$; subscripts name the inputs |
| covariance | a property of a random variable, $`\boldsymbol{\Sigma}_x`$; a kernel matrix is a covariance only for a GP prior |
| inducing inputs, inducing values | $`\mathbf{X}_{\mathcal{I}}`$ and $`\mathbf{f}_{\mathcal{I}} = f(\mathbf{X}_{\mathcal{I}})`$; "inducing points" means the inputs |
| prior, variational distribution, posterior | $`p(\mathbf{f}_{\mathcal{I}})`$, $`q(\mathbf{f}_{\mathcal{I}}; \boldsymbol{\nu})`$, $`p(\mathbf{f}_{\mathcal{I}} \mid \mathbf{y})`$ |

| Object | Write | Shape |
| --- | --- | --- |
| hyperparameters, components of $`\boldsymbol{\theta}`$ | kernel variance $`\sigma_k^2`$, noise variance $`\sigma_\varepsilon^2`$, lengthscale $`\ell`$ or $`\boldsymbol{\ell}`$ (ARD) | $`\mathbb{R}_{>0}`$, $`\mathbb{R}^{D_x}_{>0}`$ |
| named kernels, composition | $`k_\text{RBF}`$, $`k_\text{Matérn-3/2}`$; $`k_1 + k_2`$, $`k_1 k_2`$ | — |
| stationary, isotropic | $`k(\mathbf{x} - \mathbf{x}')`$, $`k(\Vert \mathbf{x} - \mathbf{x}' \Vert_2)`$ | — |
| function values | $`\mathbf{f} = f(\mathbf{X})`$, $`\mathbf{f}_* = f(\mathbf{X}_*)`$ | $`\mathbb{R}^{N}`$, $`\mathbb{R}^{M}`$ |
| kernel column | $`\mathbf{k}_{\mathcal{I}n}`$, column $`n`$ of $`\mathbf{K}_{\mathcal{I}X}`$ | $`\mathbb{R}^{N_{\mathcal{I}}}`$ |
| matrix-valued kernel, its Gram matrix | $`\mathbf{K}(\mathbf{x}, \mathbf{x}')`$, $`\mathbf{K}_{XX}`$ | $`\mathbb{R}^{D_y \times D_y}`$, $`\mathbb{R}^{N D_y \times N D_y}`$ |
| inducing set | $`\mathcal{I}`$; $`\mathbf{X}_{\mathcal{I}}`$, $`\mathbf{f}_{\mathcal{I}}`$; $`\mathbf{K}_{\mathcal{I}\mathcal{I}}`$, $`\mathbf{K}_{X\mathcal{I}}`$ | $`\mathbb{R}^{N_{\mathcal{I}} \times D_x}`$, $`\mathbb{R}^{N_{\mathcal{I}}}`$ |
| low-rank (Nyström) matrix | $`\mathbf{Q}_{XX} = \mathbf{K}_{X\mathcal{I}} \mathbf{K}_{\mathcal{I}\mathcal{I}}^{-1} \mathbf{K}_{\mathcal{I}X}`$ | $`\mathbb{R}^{N \times N}`$ |
| variational distribution | $`q(\mathbf{f}_{\mathcal{I}}; \boldsymbol{\nu}) = \mathcal{N}(\boldsymbol{\mu}_{\mathcal{I}}, \boldsymbol{\Sigma}_{\mathcal{I}})`$, $`\boldsymbol{\nu} = (\mathbf{X}_{\mathcal{I}}, \boldsymbol{\mu}_{\mathcal{I}}, \mathbf{L}_{\mathcal{I}})`$ | — |
| Cholesky factor | $`\mathbf{L}_{\mathbf{A}}`$, lower, $`\mathbf{L}_{\mathbf{A}} \mathbf{L}_{\mathbf{A}}^\top = \mathbf{A}`$; $`\mathbf{L}_{\mathcal{I}\mathcal{I}}`$ for $`\mathbf{K}_{\mathcal{I}\mathcal{I}}`$ | shape of $`\mathbf{A}`$ |
| whitened inducing values | $`\mathbf{f}_{\mathcal{I}} = \mathbf{L}_{\mathcal{I}\mathcal{I}} \mathbf{z}_{\mathcal{I}}`$, $`\mathbf{z}_{\mathcal{I}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`$ | $`\mathbb{R}^{N_{\mathcal{I}}}`$ |
| per-point marginal | $`q(f_n) = \mathcal{N}(\mu_n, \sigma_n^2)`$ | scalars |

- $`\ell`$ is only ever a lengthscale. Separable space–time kernels give $`\mathbf{K} = \mathbf{K}_{SS} \otimes \mathbf{K}_{tt}`$ when the stacked vector runs over space in the outer loop; state the order.
- Because the prior covariance is $`\mathbf{K}_{\mathcal{I}\mathcal{I}}`$, $`\boldsymbol{\mu}_{\mathcal{I}}`$ and $`\boldsymbol{\Sigma}_{\mathcal{I}}`$ always belong to $`q`$. Natural parameters are written in precision form, $`(\boldsymbol{\Lambda}_{\mathcal{I}} \boldsymbol{\mu}_{\mathcal{I}}, -\frac{1}{2} \boldsymbol{\Lambda}_{\mathcal{I}})`$.
- Titsias' $`\mathbf{Z}`$, $`\mathbf{u}`$ and Hensman's $`\mathbf{m}`$, $`\mathbf{S}`$ are quoted only when citing.
- The spectral density is the Fourier transform of the kernel (`functions-and-calculus.md`).

```math
\mathcal{L}_\text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\nu}) = -\frac{N}{N_\text{batch}} \sum_{n \in \mathcal{D}^{(i)}} \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n; \boldsymbol{\theta})] + \mathrm{KL}[q(\mathbf{f}_{\mathcal{I}}; \boldsymbol{\nu}) \Vert p(\mathbf{f}_{\mathcal{I}}; \boldsymbol{\theta})]
```

```math
\mu_n = \mathbf{k}_{\mathcal{I}n}^\top \mathbf{K}_{\mathcal{I}\mathcal{I}}^{-1} \boldsymbol{\mu}_{\mathcal{I}}
```

```math
\sigma_n^2 = k(\mathbf{x}_n, \mathbf{x}_n) - \mathbf{k}_{\mathcal{I}n}^\top \mathbf{K}_{\mathcal{I}\mathcal{I}}^{-1} (\mathbf{K}_{\mathcal{I}\mathcal{I}} - \boldsymbol{\Sigma}_{\mathcal{I}}) \mathbf{K}_{\mathcal{I}\mathcal{I}}^{-1} \mathbf{k}_{\mathcal{I}n}
```

## State-space (SDE) kernels

- A stationary kernel with a rational spectral density has a linear SDE form; its symbols are in `probability.md` ("Linear SDEs and state-space GPs") and its measurement row is $`\mathbf{H}`$, so $`k(\tau) = \mathbf{H} \exp(\mathbf{F}_\text{c} \tau) \boldsymbol{\Sigma}_{x,\infty} \mathbf{H}^\top`$ for $`\tau \ge 0`$.
- The Matérn rate is $`\lambda = \sqrt{2\nu} / \ell`$, the Särkkä–Solin letter. This is the one allowed use of $`\lambda`$ outside wavelength, on pages with no wavelength.
- A periodic kernel's period is $`T_\text{period}`$, because $`T`$ is the number of steps.
