# Functions, operators and calculus

Functions versus linear maps versus operators, derivatives, integration, and neural operators.

## Functions, operators and estimates

**Functions and linear maps.**

| Object | Write | Type or shape |
| --- | --- | --- |
| function of vectors, possibly nonlinear | $`f`$, $`h`$, $`g`$ (lowercase italic) | $`f: \mathbb{R}^{D_x} \times \mathbb{R}^{D_u} \to \mathbb{R}^{D_x}`$ |
| linear map, stored or matrix-free | $`\mathbf{F}`$, $`\mathbf{H}`$ (bold capital) | $`\mathbb{R}^{D_x \times D_x}`$, $`\mathbb{R}^{D_y \times D_x}`$ |
| linearisation at a named point | $`\mathbf{H}_k = \mathbf{J}_h(\mathbf{x}_{k \mid k-1})`$ | $`\mathbb{R}^{D_y \times D_x}`$ |
| adjoint of a real linear map | $`\mathbf{H}^\top`$ | $`\mathbb{R}^{D_x \times D_y}`$ |
| operator on a field (input is a function) | $`\mathcal{F}[x]`$, $`\mathcal{H}[x]`$ | discretises to $`f`$, $`h`$ |
| encoder, decoder | $`\mathbf{z} = g_\text{enc}(\mathbf{x}; \boldsymbol{\nu})`$, $`\hat{\mathbf{x}} = g_\text{dec}(\mathbf{z}; \boldsymbol{\theta})`$ | $`\mathbb{R}^{D_x} \to \mathbb{R}^{D_z}`$, $`\mathbb{R}^{D_z} \to \mathbb{R}^{D_x}`$ |
| basis functions, basis (design) matrix, feature vector | $`\phi_d(\mathbf{s})`$, $`\boldsymbol{\Phi}`$, $`\boldsymbol{\phi}(\mathbf{x})`$ | $`\Omega \to \mathbb{R}`$, $`\mathbb{R}^{N_{\Omega_x} \times D_z}`$, $`\mathbb{R}^{D_z}`$ |

- Pairs: dynamics $`f`$ and $`\mathbf{F}`$; observation $`h`$ and $`\mathbf{H}`$. The observation operator keeps $`h`$ and $`\mathbf{H}`$ because every alternative ($`g`$, $`\mathbf{A}`$, $`F`$, $`\mathbf{K}`$) collides with an agreed role.
- A basis is the linear decoder: $`x(\mathbf{s}) = \sum_{d=1}^{D_z} z_d \phi_d(\mathbf{s})`$ discretises to $`\mathbf{x} = \boldsymbol{\Phi} \mathbf{z}`$ with $`[\boldsymbol{\Phi}]_{nd} = \phi_d(\mathbf{s}_n)`$, $`\mathbf{s}_n \in \Omega_x`$. The linear encoder is $`\boldsymbol{\Phi}^\top`$ for an orthonormal basis (EOFs, PCA) and $`\boldsymbol{\Phi}^{+}`$ otherwise. Kernel features fit the same pattern: $`k(\mathbf{x}, \mathbf{x}') \approx \boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{x}')`$.
- Calligraphic letters cover sets, functionals ($`\mathcal{J}`$, $`\mathcal{L}`$), distributions ($`\mathcal{N}`$, $`\mathcal{GP}`$) and field operators, one role per letter. Field operators take square brackets.
- Signatures use $`\to`$, rules $`\mapsto`$, composition $`\circ`$: $`h \circ f`$.

**Derivatives.**

- Jacobian rows are outputs: $`\mathbf{J}_f = \partial f / \partial \mathbf{x} \in \mathbb{R}^{D_\text{out} \times D_\text{in}}`$.
- The gradient is a column, $`\nabla_{\boldsymbol{\theta}} \mathcal{L} \in \mathbb{R}^{D_\theta}`$.
- A Hessian never gets a letter. Write it with its variable, $`\nabla^2_{\boldsymbol{\theta}} \mathcal{L} \in \mathbb{R}^{D_\theta \times D_\theta}`$, or by its meaning: at a 3D-Var optimum it is the posterior precision $`\boldsymbol{\Lambda}_{x \mid y} = \nabla^2_{\mathbf{x}} \mathcal{J}(\mathbf{x}^\star)`$. Bare $`\nabla^2`$ is the Laplacian.
- $`\partial_t`$ is allowed in PDEs; $`\dot{\mathbf{x}} = \mathrm{d}\mathbf{x} / \mathrm{d}t`$ is only a time derivative.

**Estimates and decorations.**

| Mark | Means | Example |
| --- | --- | --- |
| $`\boldsymbol{\mu}`$ | mean of a distribution | $`\boldsymbol{\mu}_x`$ |
| bar | sample average | $`\bar{\mathbf{x}}_k = \frac{1}{N_{\mathcal{E}}} \sum_{e \in \mathcal{E}} \mathbf{x}^{(e)}_k`$ |
| hat | estimate from data; the text names the estimator | $`\hat{\mathbf{x}}_{k \mid k}`$, $`\hat{\boldsymbol{\Sigma}}_{x,k}`$ |
| tilde | perturbation or anomaly | $`\tilde{\mathbf{x}}^{(e)}_k = \mathbf{x}^{(e)}_k - \bar{\mathbf{x}}_k`$ |
| prime | a second argument only | $`k(\mathbf{x}, \mathbf{x}')`$ |

## Differential operators

| Object | Write | Shape |
| --- | --- | --- |
| gradient, always naming its variable | $`\nabla_{\boldsymbol{\theta}} \mathcal{L}`$, $`\nabla_{\mathbf{s}} x(\mathbf{s}, t)`$ | $`\mathbb{R}^{D_\theta}`$, $`\mathbb{R}^{D_s}`$ |
| divergence, curl, Laplacian of $`\mathbf{v}(\mathbf{s}) \in \mathbb{R}^{D_s}`$ or $`x(\mathbf{s})`$ | $`\nabla_{\mathbf{s}} \cdot \mathbf{v}`$, $`\nabla_{\mathbf{s}} \times \mathbf{v}`$, $`\nabla^2_{\mathbf{s}} x`$ | $`\mathbb{R}`$, $`\mathbb{R}^3`$, $`\mathbb{R}`$ |
| partial derivatives | $`\partial_t x`$, $`\partial_{s_d} x`$, or $`\frac{\partial x}{\partial t}`$ | shape of $`x`$ |
| material derivative | $`\frac{\mathrm{D} x}{\mathrm{D} t} = \partial_t x + \mathbf{v} \cdot \nabla_{\mathbf{s}} x`$ | shape of $`x`$ |
| functional derivative | $`\frac{\delta \mathcal{J}}{\delta x}(\mathbf{s})`$ | $`\mathbb{R}`$ per point |
| linearised increment | $`\delta \mathbf{x}`$ | $`\mathbb{R}^{D_x}`$ |
| adjoint variable or Lagrange multiplier of $`\mathbf{x}_k`$ | $`\mathbf{x}^\dagger_k`$ | $`\mathbb{R}^{D_x}`$ |

- A field page may declare $`\nabla := \nabla_{\mathbf{s}}`$ once; bare $`\nabla`$, $`\nabla \cdot`$, $`\nabla \times`$ and $`\nabla^2`$ are then spatial.
- $`\Delta`$ is always a difference or increment ($`\Delta t_k = t_k - t_{k-1}`$, $`\Delta \tau`$), never the Laplacian.
- Spatial directions are $`s_d`$, never `\partial_x`, which reads as a derivative with respect to the state.
- $`\delta \mathbf{x}`$ is a linearised increment (incremental 4D-Var, tangent linear); $`\tilde{\mathbf{x}}`$ stays the anomaly from a sample mean.
- A multiplier has the shape of the variable it constrains. In strong-constraint 4D-Var with $`\mathbf{x}_k = f(\mathbf{x}_{k-1})`$ and $`\mathbf{F}_k = \mathbf{J}_f(\mathbf{x}_{k-1})`$, the adjoint recursion is $`\mathbf{x}^\dagger_{k-1} = \mathbf{F}_k^\top \mathbf{x}^\dagger_k + \nabla_{\mathbf{x}_{k-1}} \mathcal{J}_{k-1}`$, with $`\nabla_{\mathbf{x}_0} \mathcal{J} = \mathbf{x}^\dagger_0`$.

## Integration and measures

- The differential is upright and last, with no spacing macro before it: $`\int_\Omega x(\mathbf{s}) \mathrm{d}\mathbf{s}`$. It is bold when the variable is a vector.
- The domain always sits under the integral sign, using the axis domains: $`\int_\Omega`$, $`\int_{\mathcal{T}}`$, $`\int_{\mathcal{S}}`$, $`\int_{t_{k-1}}^{t_k}`$. Nested integrals close innermost-first: $`\int_{\mathcal{T}} \int_\Omega \ldots \mathrm{d}\mathbf{s} \mathrm{d}t`$.
- Against a density, write the expectation $`\mathbb{E}_{p(\mathbf{x})}[g(\mathbf{x})]`$; use $`\int g(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x}`$ only when the density form is the point.
- Densities are with respect to Lebesgue measure. When a measure is needed, write $`\mathbb{P}(\mathrm{d}\mathbf{x})`$. For a continuous set, $`\vert \Omega \vert`$ is its measure (area, duration).
- Quadrature: $`\int_\Omega x(\mathbf{s}) \mathrm{d}\mathbf{s} \approx \sum_{n=1}^{N_{\Omega_x}} w_n x(\mathbf{s}_n)`$ with weights $`\mathbf{w} \in \mathbb{R}^{N_{\Omega_x}}_{>0}`$. Monte Carlo: $`\mathbb{E}_{p(\mathbf{x})}[g(\mathbf{x})] \approx \frac{1}{N_{\mathcal{E}}} \sum_{e \in \mathcal{E}} g(\mathbf{x}^{(e)})`$, importance weights $`w^{(e)}`$.
- Dynamics keep one letter with the discretisation in the subscript. Continuous: $`\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t); \boldsymbol{\theta})`$. One step: $`\mathbf{x}_k = f_{\Delta t_k}(\mathbf{x}_{k-1}, \mathbf{u}_k; \boldsymbol{\theta})`$. Name the solver $`\mathrm{ODESolve}(f, \mathbf{x}_{k-1}, t_{k-1}, t_k; \boldsymbol{\theta})`$ when it matters.

## Operators and neural operators

**Vocabulary.**

| Word | Maps | Write |
| --- | --- | --- |
| function | vector to vector | $`f(\mathbf{x})`$: lowercase italic, parentheses |
| linear map | vector to vector, linearly | $`\mathbf{F} \mathbf{x}`$: bold capital, stored or matrix-free |
| operator | function to function | $`\mathcal{F}[x]`$: calligraphic, square brackets |
| functional | function to scalar | $`\mathcal{J}[x] \in \mathbb{R}`$; on vectors $`\mathcal{J}(\mathbf{x})`$ |
| learned map, learned operator with no role | as above, with parameters | $`g_\text{label}(\mathbf{x}; \boldsymbol{\theta})`$, $`\mathcal{G}[u; \boldsymbol{\theta}]`$ |
| kernel integral operator | function to function | $`\mathcal{K}[z](\mathbf{s}) = \int_\Omega k(\mathbf{s}, \mathbf{s}') z(\mathbf{s}') \mathrm{d}\mathbf{s}'`$ |
| Fourier transform | function to function on $`\mathcal{W}_s`$ | $`\mathfrak{F}[z](\boldsymbol{\kappa})`$, inverse $`\mathfrak{F}^{-1}`$ |

- An operator with a role keeps the role letter and adds parameters: a learned emulator is $`\mathcal{F}[x; \boldsymbol{\theta}]`$, a learned observation operator $`\mathcal{H}[x; \boldsymbol{\theta}]`$. $`\mathcal{G}`$ and $`g`$ are only for learned operators and maps with no role.
- Function spaces are named after their variable: $`u \in \mathcal{U}`$, $`x \in \mathcal{X}`$, $`y \in \mathcal{Y}`$, $`z \in \mathcal{Z}`$. A neural operator is $`\mathcal{G}: \mathcal{U} \to \mathcal{X}`$, evaluated as $`\mathcal{G}[u; \boldsymbol{\theta}](\mathbf{s})`$ or on a grid as $`\mathcal{G}[u; \boldsymbol{\theta}](\mathbf{S}_x)`$.
- Sensors, where an input function is sampled, are a discrete set $`\Omega_u \subset \Omega`$ with $`N_{\Omega_u}`$ points.
- Hidden fields are latents with a layer superscript, $`z^{(l)}(\mathbf{s}) \in \mathbb{R}^{N_{\mathcal{V}_z}}`$, $`l = 0, \ldots, N_\text{layer}`$; per-point channels are variables per point, so on a grid $`D_z = N_{\Omega_x} N_{\mathcal{V}_z}`$. Layer operators are $`\mathcal{G}^{(l)}`$ and weights $`\mathbf{W}^{(l)}`$; the activation $`\sigma`$ is declared per page.
- Building blocks reuse `functions-and-calculus.md`: lifting and projection are a pointwise $`g_\text{enc}`$ and $`g_\text{dec}`$; a DeepONet branch is an encoder and its trunk a learned basis $`\boldsymbol{\phi}(\mathbf{s}; \boldsymbol{\theta})`$.
- The Fourier transform is fraktur because $`\mathcal{F}`$ is the field dynamics and a hat is an estimate. A kernel's spectral density is $`\mathfrak{F}[k](\boldsymbol{\kappa})`$.
- The adjoint of a linear operator is $`\mathcal{K}^\dagger`$, of a real matrix $`\mathbf{F}^\top`$; never `^*`. The identity operator is $`\mathrm{Id}`$, since $`\mathcal{I}`$ is the inducing set.

DeepONet is an encoder plus a learned basis:

```math
\mathbf{z} = g_\text{enc}(u(\mathbf{S}_u); \boldsymbol{\theta})
```

```math
x(\mathbf{s}) = \mathcal{G}[u; \boldsymbol{\theta}](\mathbf{s}) = \boldsymbol{\phi}(\mathbf{s}; \boldsymbol{\theta})^\top \mathbf{z}
```

A Fourier neural operator emulating the dynamics, with $`x_k(\mathbf{s}) = x(\mathbf{s}, t_k)`$:

```math
x_k = \mathcal{F}[x_{k-1}; \boldsymbol{\theta}] = g_\text{dec} \circ \mathcal{G}^{(N_\text{layer}-1)} \circ \ldots \circ \mathcal{G}^{(0)} \circ g_\text{enc}[x_{k-1}]
```

```math
\mathcal{G}^{(l)}[z] = \sigma\left(\mathbf{W}^{(l)} z + \mathfrak{F}^{-1}\left[\mathbf{R}^{(l)} \odot \mathfrak{F}[z]\right]\right)
```

Here $`\mathbf{W}^{(l)} \in \mathbb{R}^{N_{\mathcal{V}_z} \times N_{\mathcal{V}_z}}`$ and $`\mathbf{R}^{(l)}(\boldsymbol{\kappa}) \in \mathbb{C}^{N_{\mathcal{V}_z} \times N_{\mathcal{V}_z}}`$ on the retained wavenumbers $`\boldsymbol{\kappa} \in \mathcal{W}_{s,x}`$.
