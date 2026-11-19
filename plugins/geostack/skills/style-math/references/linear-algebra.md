# Sets, norms and linear algebra

## Sets, spaces and norms

| Object | Write | Notes |
| --- | --- | --- |
| norm | $\Vert \mathbf{x} \Vert_2$, $\Vert \mathbf{A} \Vert_F$ | always `\Vert`, which survives GitHub tables; always name the norm |
| weighted norm | $\Vert \mathbf{r} \Vert^2_{\mathbf{W}} = \mathbf{r}^\top \mathbf{W} \mathbf{r}$ | the subscript is the weight matrix |
| inner product | $\mathbf{a}^\top \mathbf{b}$; $\langle f, g \rangle_{\mathcal{V}}$ | angle brackets only for function spaces, with the space named |
| determinant, trace | $\det \boldsymbol{\Sigma}$, $\mathrm{tr}(\mathbf{A})$ | never `\vert \Sigma \vert`, which is set size |
| Kronecker, element-wise | $\mathbf{A} \otimes \mathbf{B}$, $\mathbf{A} \odot \mathbf{B}$ | |
| builders | $\mathrm{diag}(\mathbf{v})$, $\mathrm{vec}(\mathbf{A})$, $\mathbf{A}^{+}$ | vector to diagonal matrix, column stacking, pseudo-inverse |
| identity, zeros, ones | $\mathbf{I}_{D_x}$, $\mathbf{0}_{D_x}$, $\mathbf{1}_{D_x}$ | subscript is the size, dropped when obvious |
| definiteness | $\boldsymbol{\Sigma} \succ 0$, $\boldsymbol{\Sigma} \succeq 0$ | |
| definition | $:=$ | |
| set-builder, interval, size | $`\{\mathbf{s} \in \Omega : x(\mathbf{s}) > 0\}`$, $[a, b)$, $\vert A \vert$ | colon, never a bar |
| function spaces | $L^2(\Omega)$, RKHS $\mathcal{H}_k$ | the RKHS always carries its kernel; the field operator $\mathcal{H}[x]$ always has brackets |
| indicator | $`\mathbf{1}[x > 0] \in \{0, 1\}`$ | `\mathbb{1}` is a plain 1 in KaTeX; the ones vector never takes brackets |

**One parameterisation per page.** A cost, density or derivation works either in covariances or in precisions and does not mix them.
In covariance form the weight is an inverse covariance; in precision form it is the precision itself:

```math
\mathcal{J}(\mathbf{x}; \boldsymbol{\theta}) = \frac{1}{2} \Vert \mathbf{y} - h(\mathbf{x}; \boldsymbol{\theta}) \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}} + \frac{1}{2} \Vert \mathbf{x} - \boldsymbol{\mu}_x \Vert^2_{\boldsymbol{\Sigma}_x^{-1}}
```

```math
\mathcal{J}(\mathbf{x}; \boldsymbol{\theta}) = \frac{1}{2} \Vert \mathbf{y} - h(\mathbf{x}; \boldsymbol{\theta}) \Vert^2_{\boldsymbol{\Lambda}_\varepsilon} + \frac{1}{2} \Vert \mathbf{x} - \boldsymbol{\mu}_x \Vert^2_{\boldsymbol{\Lambda}_x}
```

Here $\mathbf{x}, \boldsymbol{\mu}_x \in \mathbb{R}^{D_x}$, $\mathbf{y} \in \mathbb{R}^{D_y}$, the observation weights are $D_y \times D_y$ and the state weights $D_x \times D_x$.
