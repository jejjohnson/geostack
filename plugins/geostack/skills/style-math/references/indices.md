# Indices, axes and groupings

Every index is declared in the text where it first appears. Sizes, axes, index defaults, and the words for samples, members and pieces of sets.

## Indices and superscripts

Every index is declared in the text where it first appears: its letter, its range and what it counts.

**Sizes.** $D$ is the length of a vector, indexed by $d$: $x_d$, $d = 1, \ldots, D_x$.
A count is the size of a discrete set, and its subscript is that set, $N_A = \vert A \vert$.
The set's own subscript names the variable that lives on it, which tells a state grid from an observation point cloud.

**Axes.** Every fundamental axis has a coordinate, a continuous domain, discrete sets, a count and an index.

| Axis | Coordinate | Domain | Discrete sets | Count | Index |
| --- | --- | --- | --- | --- | --- |
| space | $\mathbf{s} \in \mathbb{R}^{D_s}$ | $\Omega$ | $\Omega_x$ (state grid or mesh), $\Omega_y$ (observation point cloud) | $N_{\Omega_x}$ | $n$ |
| time | $t \in \mathbb{R}$ | $\mathcal{T}$ | $\mathcal{T}_x$ (model steps), $\mathcal{T}_y$ (observation times) | $N_{\mathcal{T}_x}$ | $t$ or $k$ |
| spectral | wavelength $\lambda \in \mathbb{R}_{>0}$ | $\mathcal{S}$ | $\mathcal{S}_y$ (instrument channels) | $N_{\mathcal{S}_y}$ | $c$ |
| vertical | a component of $\mathbf{s}$ ($D_s = 3$), or pressure $p \in \mathbb{R}_{>0}$ (`geoscience.md`) | part of $\Omega$, or $\mathcal{P}$ | $\mathcal{P}_x$ (model levels) | $N_{\mathcal{P}_x}$ | $l$ |
| Fourier, space | wavenumber $\boldsymbol{\kappa} \in \mathbb{R}^{D_s}$ | $\mathcal{W}_s$ | $\mathcal{W}_{s,x}$ (wavenumbers resolved on $\Omega_x$) | $N_{\mathcal{W}_{s,x}}$ | $n$ |
| Fourier, time | angular frequency $\omega \in \mathbb{R}$ | $\mathcal{W}_t$ | $\mathcal{W}_{t,x}$ | $N_{\mathcal{W}_{t,x}}$ | $k$ |
| variables | — | — | $\mathcal{V}_x$ (e.g. CH₄, wind u, wind v) | $N_{\mathcal{V}_x}$ | $v$ |
| ensemble | — | — | $\mathcal{E}$ (members or Monte Carlo draws) | $N_{\mathcal{E}}$ | $e$ |
| training data | — | — | $\mathcal{D}$ (independent pairs) | $N$ | $n$ |
| test inputs | — | — | — | $M$ | $m$ |

A vector's size is the product over the axes it spans: $D_x = N_{\Omega_x} N_{\mathcal{V}_x}$; a hyperspectral observation has $D_y = N_{\Omega_y} N_{\mathcal{S}_y}$, with entries $y_{n,c}$.
Stacked coordinates are $\mathbf{S}_x \in \mathbb{R}^{N_{\Omega_x} \times D_s}$.
Wavenumber is $\boldsymbol{\kappa}$ so that $k$ stays a time index and a kernel. Viewing and solar angles are inputs in $\mathbf{u}$, not an axis.
Wavelength owns $\lambda$, so eigenvalues of a covariance are written as the variances along each direction, $\boldsymbol{\Sigma}_x = \boldsymbol{\Phi} \mathrm{diag}(\boldsymbol{\sigma}^2) \boldsymbol{\Phi}^\top$ with $\boldsymbol{\sigma}^2 \in \mathbb{R}^{D_z}_{\ge 0}$, and general eigenvalues as $\gamma_d$.
Two sets for one variable get a word label: $\Omega_{x,\text{coarse}}$, $\Omega_{x,\text{fine}}$.
Explicit names are the default. A page with one set of a kind may declare a short form once, such as $N_\Omega \equiv N_{\Omega_x}$ or $T \equiv N_{\mathcal{T}_x}$.

**Index defaults.** They are chosen so they never collide.

| Counts | Write | Declare as |
| --- | --- | --- |
| time step, regular spacing | $\mathbf{x}_t$ | "$t$ indexes the model time steps" |
| time step, real time stamps | $\mathbf{x}_k = \mathbf{x}(t_k)$ | "$t_k \in \mathcal{T}_x$ is the $k$-th time stamp" |
| parameter (outer) iteration | $\boldsymbol{\theta}^{(i)}$, $i = 0, 1, \ldots$ | "$i$ counts optimiser iterations" |
| state (inner) iteration | $\mathbf{x}^{(j)}$, $j = 0, 1, \ldots$ | "$j$ counts inner solver iterations" |
| ensemble member | $\mathbf{x}^{(e)}$, $e \in \mathcal{E}$ | |
| sample, test sample, component | $\mathbf{x}_n$, $\mathbf{x}_m$, $x_d$ | |

Use real time stamps ($k$ and $t_k$) whenever steps are irregular or observations fall on some steps only; integrals then read $\int_{t_{k-1}}^{t_k} F(\mathbf{x}(t)) \mathrm{d}t$ and sums $\sum_{k: t_k \in \mathcal{T}_y}$.

**Order and position.**

- Subscripts are positions: variable label, then time or sample, then component. $x^{(e)}_{t,d}$ is component $d$ of member $e$ at time $t$; $\boldsymbol{\Sigma}_{x,t}$.
- Parenthesised superscripts are copies of a variable (members, iterations). Matrix entries $[\mathbf{A}]_{ij}$ keep $i, j$ as subscripts inside brackets.
- What an estimate is conditioned on replaces forecast and analysis superscripts: forecast $t \mid t-1$, analysis $t \mid t$, smoothed $t \mid T$ (or $k \mid k-1$, …).
- $_*$ is a test point ($\mathbf{x}_*$, $`\mathbf{K}_{*X}`$); $^\star$ is an optimum ($\boldsymbol{\theta}^\star$). Never `^*`, which reads as a conjugate or adjoint.
- Ranges: $\mathbf{x}_{1:T} \in \mathbb{R}^{T \times D_x}$ for a run; $\mathbf{x}_0$ is the initial state, so $\mathbf{x}_{0:T} \in \mathbb{R}^{(T+1) \times D_x}$; $n = 1, \ldots, N$ in prose with `\ldots`.

## Samples, members, batches, patches and blocks

**Vocabulary.** Use these words with these meanings, in prose and in code names.

| Word | Meaning | Write | Count |
| --- | --- | --- | --- |
| sample | one observed data pair | $(\mathbf{x}_n, \mathbf{y}_n) \in \mathcal{D}$ | $N = \vert \mathcal{D} \vert$ |
| member | one draw from a distribution or model: ensemble member, Monte Carlo draw, deep-ensemble model $\boldsymbol{\theta}^{(e)}$ | $\mathbf{x}^{(e)}$, $e \in \mathcal{E}$ | $N_{\mathcal{E}}$ |
| batch | the samples used at optimiser iteration $i$ | $\mathcal{D}^{(i)} \subset \mathcal{D}$ | $N_\text{batch}$ |
| tile | a piece of a partition of space; tiles cover $\Omega_x$ without overlap | $\Omega_x^{(r)}$ | $N_\text{tile}$ |
| patch | a sub-region of space that may overlap others (training crop, local analysis domain) | $\Omega_x^{(p)} \subset \Omega_x$ | $N_\text{patch}$ |
| block | a piece of a partition of time (extremes blocks) | $\mathcal{T}_y^{(b)}$ | $N_\text{block}$ |
| window | a time interval that may overlap others (assimilation window, sliding window) | $\mathcal{T}_x^{(w)} \subset \mathcal{T}_x$ | $N_\text{window}$ |

- An element of a set takes a position subscript or a copy superscript: $\mathbf{x}_n$, $\mathbf{x}^{(e)}$.
- A piece of a set is the set with a parenthesised superscript naming the piece; the letter says which kind ($r$ tile, $p$ patch, $b$ block, $w$ window). A variable restricted to a piece takes the same superscript, $\mathbf{x}^{(p)} \in \mathbb{R}^{\vert \Omega_x^{(p)} \vert N_{\mathcal{V}_x}}$; several stack as $\mathbf{x}^{(e,p)}$.
- Tile and patch in space mirror block and window in time: a partition versus pieces that may overlap.
- The number of pieces is $N$ with a word label; the size of one piece is its set size, $\vert \Omega_x^{(p)} \vert$.

```math
\mathcal{L}(\boldsymbol{\theta}) \approx \frac{1}{N_\text{batch}} \sum_{n \in \mathcal{D}^{(i)}} \frac{1}{N_\text{patch}} \sum_{p=1}^{N_\text{patch}} \Vert \mathbf{y}_n^{(p)} - g(\mathbf{x}_n^{(p)}; \boldsymbol{\theta}) \Vert_2^2
```
