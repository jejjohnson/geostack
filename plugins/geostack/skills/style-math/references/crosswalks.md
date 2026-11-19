# Crosswalks from classic letters

Papers use the letters on the left. Write the standard form; quote a classic letter only when citing a source, and say which standard symbol it maps to.

## Data assimilation

Papers and older notes use the classic letters on the left.
Write the standard form on the right; quote the classic letter only when citing a source, and say which it maps to.

| Classic | Standard | Shape | Meaning |
| --- | --- | --- | --- |
| $`\mathbf{x}_b`$, $`\mathbf{m}`$ | $`\boldsymbol{\mu}_x`$ | $`\mathbb{R}^{D_x}`$ | prior (background) mean of the state |
| $`\mathbf{B}`$, $`\mathbf{P}^f`$ | $`\boldsymbol{\Sigma}_x`$ | $`\mathbb{R}^{D_x \times D_x}`$ | prior (background, forecast) state covariance |
| $`\mathbf{R}`$ | $`\boldsymbol{\Sigma}_\varepsilon`$ | $`\mathbb{R}^{D_y \times D_y}`$ | observation-error covariance |
| $`\mathbf{Q}`$ | $`\boldsymbol{\Sigma}_\eta`$ | $`\mathbb{R}^{D_x \times D_x}`$ | model-error covariance |
| $`\mathbf{S}`$, $`\mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R}`$ | $`\boldsymbol{\Sigma}_y`$ | $`\mathbb{R}^{D_y \times D_y}`$ | covariance of the observations (innovation covariance) |
| $`\mathbf{P}\mathbf{H}^\top`$ | $`\boldsymbol{\Sigma}_{xy}`$ | $`\mathbb{R}^{D_x \times D_y}`$ | state–observation cross-covariance |
| $`\mathbf{K}`$, $`\mathbf{K}_t`$ | $`\mathbf{G}_t = \boldsymbol{\Sigma}_{xy,t} \boldsymbol{\Sigma}_{y,t}^{-1}`$ | $`\mathbb{R}^{D_x \times D_y}`$ | Kalman gain |
| $`\mathbf{x}^f`$, $`\mathbf{x}^a`$, $`\mathbf{x}^s`$ | $`\mathbf{x}_{t \mid t-1}`$, $`\mathbf{x}_{t \mid t}`$, $`\mathbf{x}_{t \mid T}`$ | $`\mathbb{R}^{D_x}`$ | forecast, analysis, smoothed state |
| $`\mathbf{B}^{-1}`$ | $`\boldsymbol{\Lambda}_x`$ | $`\mathbb{R}^{D_x \times D_x}`$ | prior precision |
| $`J(\mathbf{x})`$ | $`\mathcal{J}(\mathbf{x}; \boldsymbol{\theta})`$ | $`\mathbb{R}`$ | variational cost over the state |
| $`\mathbf{M}`$, $`\mathcal{M}`$ | $`f`$ (nonlinear), $`\mathbf{F}`$ (linear) | $`\mathbb{R}^{D_x} \to \mathbb{R}^{D_x}`$ | dynamical model |
| $`\mathbf{H}`$, $`\mathcal{H}`$ | $`\mathbf{H}`$ (linear), $`h`$ (nonlinear) | $`\mathbb{R}^{D_y \times D_x}`$, $`\mathbb{R}^{D_x} \to \mathbb{R}^{D_y}`$ | observation operator |

$`\mathbf{B}`$ is free in this standard but never used as a basis, so a DA reader does not mistake it for the background covariance; bases are $`\boldsymbol{\Phi}`$.

## Retrieval (Rodgers 2000)

Retrieval papers use Rodgers' optimal-estimation letters on the left.
Write the standard form on the right; quote the classic letter only when citing a source.

| Rodgers | Standard | Shape | Meaning |
| --- | --- | --- | --- |
| $`\mathbf{x}`$, $`\mathbf{y}`$ | $`\mathbf{x}`$, $`\mathbf{y}`$ | $`\mathbb{R}^{D_x}`$, $`\mathbb{R}^{D_y}`$ | state, measurement |
| $`\mathbf{x}_a`$ | $`\boldsymbol{\mu}_x`$ | $`\mathbb{R}^{D_x}`$ | prior mean |
| $`\mathbf{S}_a`$ | $`\boldsymbol{\Sigma}_x`$ | $`\mathbb{R}^{D_x \times D_x}`$ | prior covariance |
| $`\mathbf{S}_\epsilon`$ | $`\boldsymbol{\Sigma}_\varepsilon`$ | $`\mathbb{R}^{D_y \times D_y}`$ | measurement-error covariance |
| $`F(\mathbf{x}, \mathbf{b})`$ | $`h(\mathbf{x}; \mathbf{u})`$ | $`\mathbb{R}^{D_x} \to \mathbb{R}^{D_y}`$ | forward model; Rodgers' $`\mathbf{b}`$ are inputs $`\mathbf{u}`$ |
| $`\mathbf{K}`$ | $`\mathbf{H} = \mathbf{J}_h`$ | $`\mathbb{R}^{D_y \times D_x}`$ | weighting function (Jacobian) |
| $`\mathbf{G}`$, $`\mathbf{D}_y`$ | $`\mathbf{G}`$ | $`\mathbb{R}^{D_x \times D_y}`$ | gain (contribution function) |
| $`\mathbf{A}`$ | $`\mathbf{J}_{\hat{x}} = \mathbf{G} \mathbf{H}`$ | $`\mathbb{R}^{D_x \times D_x}`$ | averaging kernel |
| $`\hat{\mathbf{S}}`$ | $`\boldsymbol{\Sigma}_{x \mid y}`$ | $`\mathbb{R}^{D_x \times D_x}`$ | posterior covariance |

## Sparse and variational GPs

| Classic | Standard | Shape | Meaning |
| --- | --- | --- | --- |
| $`\mathbf{Z}`$ (Titsias 2009) | $`\mathbf{X}_{\mathcal{I}}`$ | $`\mathbb{R}^{N_{\mathcal{I}} \times D_x}`$ | inducing inputs |
| $`\mathbf{u}`$ | $`\mathbf{f}_{\mathcal{I}}`$ | $`\mathbb{R}^{N_{\mathcal{I}}}`$ | inducing values |
| $`\mathbf{K}_{uu}`$, $`\mathbf{K}_{fu}`$ | $`\mathbf{K}_{\mathcal{I}\mathcal{I}}`$, $`\mathbf{K}_{X\mathcal{I}}`$ | $`N_{\mathcal{I}} \times N_{\mathcal{I}}`$, $`N \times N_{\mathcal{I}}`$ | kernel blocks |
| $`\mathbf{m}`$, $`\mathbf{S}`$ (Hensman et al. 2013) | $`\boldsymbol{\mu}_{\mathcal{I}}`$, $`\boldsymbol{\Sigma}_{\mathcal{I}}`$ | $`\mathbb{R}^{N_{\mathcal{I}}}`$, $`\mathbb{R}^{N_{\mathcal{I}} \times N_{\mathcal{I}}}`$ | variational mean, covariance |
| $`\sigma_f^2`$, $`\sigma_n^2`$ | $`\sigma_k^2`$, $`\sigma_\varepsilon^2`$ | $`\mathbb{R}_{>0}`$ | kernel variance, noise variance |
| $`\mathbf{z}`$, $`\mathbf{u}`$ in VAEs | $`\mathbf{z}`$; encoder parameters $`\boldsymbol{\nu}`$, not $`\boldsymbol{\phi}`$ | $`\mathbb{R}^{D_z}`$, $`\mathbb{R}^{D_\nu}`$ | latent, encoder parameters |

## Extremes

| Classic | Standard | Meaning |
| --- | --- | --- |
| $`\kappa`$ (Hosking) | $`\xi = -\kappa`$ (Coles) | GEV and GPD shape; the sign flips |
| $`\lambda`$ | $`\rho_u`$ | exceedance rate of threshold $`u`$ |
| $`\bar{F}`$ | $`1 - F`$ | survival function |
| $`M_n`$ | $`y_{\text{max},b}`$ | block maximum |
