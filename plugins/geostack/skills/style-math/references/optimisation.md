# Optimisation

Builds on the cost $`\mathcal{J}(\mathbf{x}; \boldsymbol{\theta})`$ and loss $`\mathcal{L}(\boldsymbol{\theta})`$ of `SKILL.md`.

## Optimisation

| Object | Write | Shape |
| --- | --- | --- |
| step size on parameters (learning rate), on the state | $`\gamma`$, $`\alpha`$; schedules $`\gamma^{(i)}`$ | $`\mathbb{R}_{>0}`$ |
| regulariser and its weight | $`\mathcal{L}(\boldsymbol{\theta}) + \beta \mathcal{R}(\boldsymbol{\theta})`$ | $`\mathcal{R} \in \mathbb{R}`$, $`\beta \in \mathbb{R}_{\ge 0}`$ |
| constraints | $`c_\text{eq}(\mathbf{x}) = \mathbf{0}`$, $`c_\text{ineq}(\mathbf{x}) \le \mathbf{0}`$ | $`c: \mathbb{R}^{D_x} \to \mathbb{R}^{D_c}`$ |
| multiplier of a constraint | $`\mathbf{c}^\dagger`$ | $`\mathbb{R}^{D_c}`$ |
| Lagrangian | $`\mathcal{J}_\text{Lag}(\mathbf{x}, \mathbf{c}^\dagger) = \mathcal{J}(\mathbf{x}) + \mathbf{c}^{\dagger\top} c_\text{eq}(\mathbf{x})`$ | $`\mathbb{R}`$ |
| proximal operator | $`\mathrm{prox}_{\gamma \mathcal{R}}(\mathbf{v}) = \mathrm{argmin}_{\mathbf{x}} \ \mathcal{R}(\mathbf{x}) + \frac{1}{2\gamma} \Vert \mathbf{x} - \mathbf{v} \Vert_2^2`$ | $`\mathbb{R}^{D_x}`$ |
| partial gradient, total derivative (hypergradient) | $`\nabla_{\boldsymbol{\theta}} \mathcal{L}`$, $`\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\theta}}`$ | $`\mathbb{R}^{D_\theta}`$ |
| mixed second derivative | $`\nabla^2_{\mathbf{x}\boldsymbol{\theta}} \mathcal{J}`$, rows $`\mathbf{x}`$, columns $`\boldsymbol{\theta}`$ | $`\mathbb{R}^{D_x \times D_\theta}`$ |

- Problems are written with "s.t." in text: $`\min_{\mathbf{x}} \mathcal{J}(\mathbf{x}) \ \text{s.t.} \ c_\text{eq}(\mathbf{x}) = \mathbf{0}`$.
- Never $`\eta`$ for a step size (model noise), $`\lambda`$ for a regularisation weight (wavelength), $`\mathcal{L}`$ for a Lagrangian (loss) or $`\epsilon`$ for a tolerance (write $`\mathrm{tol}`$).
- Regression coefficients are parameters, so they are $`\boldsymbol{\theta}`$.
- Implicit differentiation of $`\mathbf{x}^\star(\boldsymbol{\theta}) = \mathrm{argmin}_{\mathbf{x}} \mathcal{J}(\mathbf{x}; \boldsymbol{\theta})`$:

```math
\frac{\partial \mathbf{x}^\star}{\partial \boldsymbol{\theta}} = -\left(\nabla^2_{\mathbf{x}} \mathcal{J}\right)^{-1} \nabla^2_{\mathbf{x}\boldsymbol{\theta}} \mathcal{J}
```

```math
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\theta}} = \nabla_{\boldsymbol{\theta}} \mathcal{L} + \left(\frac{\partial \mathbf{x}^\star}{\partial \boldsymbol{\theta}}\right)^\top \nabla_{\mathbf{x}} \mathcal{L}
```
