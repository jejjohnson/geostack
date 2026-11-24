---
name: style-math
description: Eman's maths notation standard for GitHub Markdown, MyST, mkdocs and Unicode in code — symbols with shapes, bold vs italic, kind-named macros, variable roles (x, y, z, u, θ), role-and-subscript names (Σ_x, K_XX, 𝒥, ℒ), vocabularies, and per-topic rules from probability to neural operators. Use whenever writing or reviewing maths in his notes, docs, issues or docstrings.
---

# Maths notation standard

Draft 2 · 19 Nov 2026 · built round by round with Eman; GitHub inline maths is in the backtick form throughout.

Maths reads the same in every output format: the letter says what kind of object a symbol is, the subscript says which variable it belongs to, and every symbol comes with its shape.
This file holds the principles, the core symbols, a quick reference and the vocabularies; each topic's full rules are in `references/`.
How a symbol becomes a variable name or a dim name in code is in `style-python` (§3 and §6), and pseudocode is in `style-code-sketch`; neither is repeated here.

## How to use it

1. Write with the macros, giving every symbol its shape.
2. Check the file against the standard and the target renderer. The scripts are in this skill's `scripts/` folder; `<skill folder>` below is the base directory given when the skill loads. Each script declares its dependencies (cyclopts, loguru, pyyaml) inline, so `uv run` installs them; plain `python` works only where they are already installed:

```bash
uv run <skill folder>/scripts/check_math.py notes.md --target myst
```

3. For GitHub, run the expander, then check again. It expands the macros and rewrites every inline `$…$` in the backtick form, without which Markdown pairs the underscores of two expressions as emphasis and both are lost. Run it on a file written without macros too:

```bash
uv run <skill folder>/scripts/expand_macros.py body.md
```

```bash
uv run <skill folder>/scripts/check_math.py body.md --target github
```

4. For a MyST or mkdocs project, install the macros once with `scripts/emit_macros.py` (`references/targets.md`).

The checker cannot see shapes, roles or one-meaning-per-page; review those by hand with the checklist.

## 1. Principles

1. **Every symbol has a shape.** Give it next to every equation, as `\mathbf{x} \in \mathbb{R}^{D_x}` or in a shape list below the block. Name each dimension once per page.
2. **Bold follows the shape.** Scalars are plain italic, vectors and matrices are bold, sets are never bold. A single-test-point GP mean $`\mu_*`$ is a scalar, so it is not bold.
3. **One letter per role.** The letter names the kind of object (covariance, kernel, cost); the subscript names the variable it belongs to. A letter is never reused for a second role on the same page.
4. **Position, not knowledge, decides the letter.** In $`\mathbf{y} = f(\mathbf{x})`$, $`\mathbf{x}`$ is the input and $`\mathbf{y}`$ the output whether $`\mathbf{x}`$ is known (regression) or inferred (inversion). The prose says which.
5. **Write macros, not styles.** Use the kind-named macros in MyST and mkdocs; GitHub gets the expanded form.

## 2. Bold style and macros

Latin vectors and matrices are upright bold (`\mathbf`); Greek ones are bold italic (`\boldsymbol`).
This is the Rasmussen & Williams and Bishop convention.

| Kind | Latin | Greek | Renders |
| --- | --- | --- | --- |
| vector | `\vect{x}` | `\gvect{\mu}` | $`\mathbf{x}`$, $`\boldsymbol{\mu}`$ |
| matrix | `\mat{K}` | `\gmat{\Sigma}` | $`\mathbf{K}`$, $`\boldsymbol{\Sigma}`$ |
| operator name | `\op{diag}` | — | $`\mathrm{diag}`$ |
| scalar | `x` | `\sigma` | $`x`$, $`\sigma`$ |

Each kind needs two macros because KaTeX, which MyST's book theme uses, cannot bold both alphabets with one definition.
`\vect{\mu}` and `\mathbf{\theta}` render unbolded; the checker flags both.
Never use `\bm` (MathJax rejects it) or `\operatorname` (GitHub rejects it).

## 3. Variable roles

| Role | Symbol | Shape | Notes |
| --- | --- | --- | --- |
| forward-model input | $`\mathbf{x}`$ | $`\mathbb{R}^{D_x}`$ | a covariate in regression, the state in inversion and data assimilation |
| forward-model output, observations | $`\mathbf{y}`$ | $`\mathbb{R}^{D_y}`$ | |
| latent | $`\mathbf{z}`$ | $`\mathbb{R}^{D_z}`$ | no direct physical meaning, usually $`D_z \ll D_x`$ |
| exogenous input | $`\mathbf{u}_t`$ | $`\mathbb{R}^{D_u}`$ | covariate, forcing or control; varies over time or sample; the page says whether it is given or optimised; split as $`\mathbf{u}^\text{dyn}`$, $`\mathbf{u}^\text{obs}`$ if needed |
| parameters | $`\boldsymbol{\theta}`$ | $`\mathbb{R}^{D_\theta}`$ | constant |
| variational or encoder parameters | $`\boldsymbol{\nu}`$ | $`\mathbb{R}^{D_\nu}`$ | as in Blei et al.'s VI review; $`\boldsymbol{\nu}`$ is the letter because $`\boldsymbol{\phi}`$ is reserved for basis functions |
| observation noise | $`\boldsymbol{\varepsilon}`$ | $`\mathbb{R}^{D_y}`$ | $`\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)`$ |
| model noise | $`\boldsymbol{\eta}`$ | $`\mathbb{R}^{D_x}`$ | $`\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\eta)`$ |
| coordinates | $`\mathbf{s}`$, $`t`$ | $`\mathbf{s} \in \Omega \subset \mathbb{R}^{D_s}`$, $`t \in \mathcal{T} \subset \mathbb{R}`$ | |
| field | $`x(\mathbf{s}, t)`$ | $`\mathbb{R}`$ | the state as a function; a physical letter ($`\psi`$, $`T`$, $`q`$) is allowed when the physics matters |

A capital letter is $`N`$ samples stacked as rows, $`\mathbf{X} \in \mathbb{R}^{N \times D_x}`$.
A range subscript is a run of time steps, $`\mathbf{x}_{1:T} \in \mathbb{R}^{T \times D_x}`$.

The standard state-space model:

```math
\mathbf{x}_t = f(\mathbf{x}_{t-1}, \mathbf{u}_t; \boldsymbol{\theta}) + \boldsymbol{\eta}_t
```

```math
\mathbf{y}_t = h(\mathbf{x}_t, \mathbf{u}_t; \boldsymbol{\theta}) + \boldsymbol{\varepsilon}_t
```

**Vocabulary.** The words for what a variable is:

| Word | Meaning | Symbol |
| --- | --- | --- |
| input | what goes into the forward model $`y = f(x)`$; a covariate in regression, the state in inversion | $`\mathbf{x}`$ |
| state | the input when it is inferred: the physical system at a time | $`\mathbf{x}_t`$, $`\mathbf{x}_k`$ |
| observation | the output of the forward model, what an instrument measured | $`\mathbf{y}`$ |
| latent | a hidden variable with no direct physical meaning | $`\mathbf{z}`$ |
| exogenous input | known or optimised, varies over time or sample, never inferred | $`\mathbf{u}`$ |
| covariate, forcing | an exogenous input that is given (to a statistical model; to the dynamics) | $`\mathbf{u}`$ |
| control | an exogenous input that is optimised | $`\mathbf{u}`$ |
| parameter | constant over time and sample | $`\boldsymbol{\theta}`$ |
| variational or encoder parameter | parameters of an approximate posterior or encoder | $`\boldsymbol{\nu}`$ |
| field | a variable as a function of space and time | $`x(\mathbf{s}, t)`$ |

## 4. Role families

| Role | Symbols | Shapes |
| --- | --- | --- |
| mean | $`\boldsymbol{\mu}_x`$ | $`\mathbb{R}^{D_x}`$ |
| covariance | $`\boldsymbol{\Sigma}_x`$, $`\boldsymbol{\Sigma}_\varepsilon`$, $`\boldsymbol{\Sigma}_\eta`$, $`\boldsymbol{\Sigma}_y`$, $`\boldsymbol{\Sigma}_{xy}`$ | $`D_x{\times}D_x`$, $`D_y{\times}D_y`$, $`D_x{\times}D_x`$, $`D_y{\times}D_y`$, $`D_x{\times}D_y`$ |
| precision | $`\boldsymbol{\Lambda}_x = \boldsymbol{\Sigma}_x^{-1}`$ | $`D_x{\times}D_x`$ |
| kernel matrix | $`\mathbf{K}_{XX}`$, $`\mathbf{K}_{*X}`$, $`\mathbf{K}_{**}`$ | $`N{\times}N`$, $`M{\times}N`$, $`M{\times}M`$ |
| gain | $`\mathbf{G}_t = \boldsymbol{\Sigma}_{xy,t} \boldsymbol{\Sigma}_{y,t}^{-1}`$ | $`D_x{\times}D_y`$ |
| smoother gain | $`\mathbf{G}_{\text{smooth},t} = \boldsymbol{\Sigma}_{x,t \mid t} \mathbf{F}_{t+1}^\top \boldsymbol{\Sigma}_{x,t+1 \mid t}^{-1}`$ | $`D_x{\times}D_x`$ |
| residual (innovation) | $`\mathbf{r}_t = \mathbf{y}_t - \mathbf{H}_t \boldsymbol{\mu}_{x,t \mid t-1}`$ | $`\mathbb{R}^{D_y}`$ |
| Jacobian | $`\mathbf{J}_h = \partial h / \partial \mathbf{x}`$ | $`D_y{\times}D_x`$ |
| cost over the state | $`\mathcal{J}(\mathbf{x}; \boldsymbol{\theta})`$ | scalar |
| loss over the parameters | $`\mathcal{L}(\boldsymbol{\theta})`$ | scalar |

The subscript names the random variable: $`\boldsymbol{\Sigma}_y = \mathbf{H} \boldsymbol{\Sigma}_x \mathbf{H}^\top + \boldsymbol{\Sigma}_\varepsilon`$ is the covariance of $`\mathbf{y}`$, not the observation error.
$`\mathbf{K}`$ is only ever a kernel matrix; its subscripts name the stacked inputs ($`\mathbf{K}_{SS}`$ when they are coordinates).
A named loss puts its name in the subscript and is always the quantity minimised: $`\mathcal{L}_\text{ELBO} = -\mathrm{ELBO}`$, $`\mathcal{L}_\text{NLL}`$, $`\mathcal{L}_\text{MSE}`$. Many papers use $`\mathcal{L}_\text{ELBO}`$ for the ELBO itself, so state the sign once on any page that uses it.
Cost and loss are calligraphic, which keeps bold $`\mathbf{J}`$ free for Jacobians and matches bilevel problems:

```math
\boldsymbol{\theta}^\star = \mathrm{argmin}_{\boldsymbol{\theta}} \ \mathcal{L}(\boldsymbol{\theta}; \mathbf{x}^\star(\boldsymbol{\theta}))
```

```math
\mathbf{x}^\star(\boldsymbol{\theta}) = \mathrm{argmin}_{\mathbf{x}} \ \mathcal{J}(\mathbf{x}; \boldsymbol{\theta})
```

## 5. Quick reference

| Topic | Rule | Example | Details |
| --- | --- | --- | --- |
| subscripts | variable label, then time or sample, then component | $`\boldsymbol{\Sigma}_{x,t \mid t-1}`$, $`x_{n,d}`$ | `indices.md` |
| superscripts | a parenthesised copy or piece; the letter says which kind | $`\mathbf{x}^{(e)}`$, $`\boldsymbol{\theta}^{(i)}`$, $`\Omega_x^{(p)}`$ | `indices.md` |
| sizes | $`D`$ is a vector's length; a count is a set's size | $`D_x = N_{\Omega_x} N_{\mathcal{V}_x}`$ | `indices.md` |
| time | $`t`$ for regular steps; $`k`$ with $`t_k`$ declared for real time stamps | $`\mathbf{x}_k = \mathbf{x}(t_k)`$ | `indices.md` |
| optimum, test point | $`^\star`$ and $`_*`$, never `^*` | $`\boldsymbol{\theta}^\star`$, $`\mathbf{x}_*`$ | `indices.md` |
| transpose, norm, conditioning | `^\top`, `\Vert`, `\mid` | $`\Vert \mathbf{r} \Vert_2`$, $`p(\mathbf{y} \mid \mathbf{x}; \boldsymbol{\theta})`$ | `linear-algebra.md`, `probability.md` |
| fixed parameters | after a semicolon, in functions and densities alike | $`f(\mathbf{x}; \boldsymbol{\theta})`$ | `probability.md` |
| operators on random variables | blackboard bold | $`\mathbb{E}`$, $`\mathbb{V}`$, $`\mathbb{P}`$, $`\mathbb{H}`$, $`\mathbb{I}`$ | `probability.md` |
| covariance or precision | one parameterisation per page | $`\Vert \mathbf{r} \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}}`$ or $`\Vert \mathbf{r} \Vert^2_{\boldsymbol{\Lambda}_\varepsilon}`$ | `linear-algebra.md` |
| functions, linear maps, operators | $`f`$; $`\mathbf{F}`$; $`\mathcal{F}[x]`$ | $`\mathbf{H}_k = \mathbf{J}_h(\mathbf{x}_{k \mid k-1})`$ | `functions-and-calculus.md` |
| derivatives | $`\nabla`$ names its variable; $`\Delta`$ is a difference; upright $`\mathrm{d}`$ | $`\nabla_{\boldsymbol{\theta}} \mathcal{L}`$, $`\int_\Omega x \mathrm{d}\mathbf{s}`$ | `functions-and-calculus.md` |
| decorations | $`\boldsymbol{\mu}`$ distribution mean, bar sample average, hat estimate, tilde anomaly, prime second argument | $`\tilde{\mathbf{x}}^{(e)} = \mathbf{x}^{(e)} - \bar{\mathbf{x}}`$ | `functions-and-calculus.md` |
| learned maps | word-labelled $`g`$; an operator with a role keeps its letter | $`g_\text{enc}`$, $`\mathcal{F}[x; \boldsymbol{\theta}]`$ | `functions-and-calculus.md` |
| losses | $`\mathcal{L}`$ is minimised; a named loss carries its name | $`\mathcal{L}_\text{ELBO} = -\mathrm{ELBO}`$ | `optimisation.md` |
| kernels | $`k`$ with two arguments; $`\mathbf{K}`$ subscripted by its inputs | $`\mathbf{K}_{X\mathcal{I}}`$ | `kernels.md` |
| physics | classic symbols, declared in a symbol table, one meaning per page | $`u(\mathbf{s}, t)`$, $`p`$, $`\rho`$ | `geoscience.md` |

## 6. Vocabularies

**Samples, members and pieces of sets** (`references/indices.md`):

| Word | Meaning | Write | Count |
| --- | --- | --- | --- |
| sample | one observed data pair | $`(\mathbf{x}_n, \mathbf{y}_n) \in \mathcal{D}`$ | $`N = \vert \mathcal{D} \vert`$ |
| member | one draw from a distribution or model: ensemble member, Monte Carlo draw, deep-ensemble model $`\boldsymbol{\theta}^{(e)}`$ | $`\mathbf{x}^{(e)}`$, $`e \in \mathcal{E}`$ | $`N_{\mathcal{E}}`$ |
| batch | the samples used at optimiser iteration $`i`$ | $`\mathcal{D}^{(i)} \subset \mathcal{D}`$ | $`N_\text{batch}`$ |
| tile | a piece of a partition of space; tiles cover $`\Omega_x`$ without overlap | $`\Omega_x^{(r)}`$ | $`N_\text{tile}`$ |
| patch | a sub-region of space that may overlap others (training crop, local analysis domain) | $`\Omega_x^{(p)} \subset \Omega_x`$ | $`N_\text{patch}`$ |
| block | a piece of a partition of time (extremes blocks) | $`\mathcal{T}_y^{(b)}`$ | $`N_\text{block}`$ |
| window | a time interval that may overlap others (assimilation window, sliding window) | $`\mathcal{T}_x^{(w)} \subset \mathcal{T}_x`$ | $`N_\text{window}`$ |

**Kernels** (`references/kernels.md`):

| Word | Meaning |
| --- | --- |
| kernel | the function $`k(\mathbf{x}, \mathbf{x}')`$, always with two arguments |
| kernel matrix (Gram matrix) | $`k`$ evaluated on stacked inputs, $`\mathbf{K}_{XX}`$; subscripts name the inputs |
| covariance | a property of a random variable, $`\boldsymbol{\Sigma}_x`$; a kernel matrix is a covariance only for a GP prior |
| inducing inputs, inducing values | $`\mathbf{X}_{\mathcal{I}}`$ and $`\mathbf{f}_{\mathcal{I}} = f(\mathbf{X}_{\mathcal{I}})`$; "inducing points" means the inputs |
| prior, variational distribution, posterior | $`p(\mathbf{f}_{\mathcal{I}})`$, $`q(\mathbf{f}_{\mathcal{I}}; \boldsymbol{\nu})`$, $`p(\mathbf{f}_{\mathcal{I}} \mid \mathbf{y})`$ |

**Functions and operators** (`references/functions-and-calculus.md`):

| Word | Maps | Write |
| --- | --- | --- |
| function | vector to vector | $`f(\mathbf{x})`$: lowercase italic, parentheses |
| linear map | vector to vector, linearly | $`\mathbf{F} \mathbf{x}`$: bold capital, stored or matrix-free |
| operator | function to function | $`\mathcal{F}[x]`$: calligraphic, square brackets |
| functional | function to scalar | $`\mathcal{J}[x] \in \mathbb{R}`$; on vectors $`\mathcal{J}(\mathbf{x})`$ |
| learned map, learned operator with no role | as above, with parameters | $`g_\text{label}(\mathbf{x}; \boldsymbol{\theta})`$, $`\mathcal{G}[u; \boldsymbol{\theta}]`$ |
| kernel integral operator | function to function | $`\mathcal{K}[z](\mathbf{s}) = \int_\Omega k(\mathbf{s}, \mathbf{s}') z(\mathbf{s}') \mathrm{d}\mathbf{s}'`$ |
| Fourier transform | function to function on $`\mathcal{W}_s`$ | $`\mathfrak{F}[z](\boldsymbol{\kappa})`$, inverse $`\mathfrak{F}^{-1}`$ |

## 7. References

| File | Covers |
| --- | --- |
| `references/probability.md` | densities and conditioning, expectations, distribution functions, extremes, stochastic processes |
| `references/indices.md` | sizes, axes (space, time, spectral, vertical, Fourier), index defaults, samples, batches, patches, tiles, blocks, windows |
| `references/functions-and-calculus.md` | functions, linear maps, bases, derivatives, integration, operators, neural operators |
| `references/linear-algebra.md` | norms, determinants, matrix products, sets, function spaces, one parameterisation per page |
| `references/optimisation.md` | step sizes, regularisation, constraints, Lagrangians, proximal operators, implicit differentiation |
| `references/kernels.md` | kernels, hyperparameters, sparse and variational GPs |
| `references/geoscience.md` | classic physics symbols, units, methane retrieval |
| `references/crosswalks.md` | classic letters from DA, retrieval, GP and extremes papers |
| `references/targets.md` | GitHub, MyST, mkdocs and Unicode syntax, with what was verified |
| `references/macros.yml` | the macro definitions |

## 8. Old notes

The standard applies to new pages and to pages you edit.
Do not rewrite old notes wholesale; they use both bold styles and older letters ($`\mathbf{B}`$, $`\mathbf{R}`$, $`\mathbf{K}_t`$ for the gain).

## Checklist

- [ ] Every symbol has a shape, and bold matches the shape.
- [ ] Every index is declared where it first appears; counts are set sizes.
- [ ] Each letter has one role on the page; physics symbols are declared in a symbol table.
- [ ] Kind-named macros in MyST and mkdocs, expanded for GitHub; Latin and Greek macros not swapped.
- [ ] One parameterisation (covariance or precision) per page.
- [ ] On GitHub, every inline expression is in the backtick form.
- [ ] `check_math.py` passes for the target.
