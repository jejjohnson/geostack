# Capability index (excerpt)

What ships today at the pinned versions. A name that is not in this excerpt has not been checked.

## gaussx (`import gaussx as gx`)

| Name | Kind | What it does |
|---|---|---|
| `solve` | function | Solve A x = b with structural dispatch on the operator type. `gx.solve(operator, vector)`. |
| `solve_matrix` | function | Solve A X = B with one factorisation for the whole matrix right-hand side. `gx.solve_matrix(operator, matrix)`. |
| `logdet` | function | Log-determinant log |det A| with structural dispatch on the operator type. `gx.logdet(operator)`. |
| `cholesky` | function | Lower Cholesky factor L of an operator, structured where A is. |

## kernellib (`import kernellib as kl`)

| Name | Kind | What it does |
|---|---|---|
| `RBF` | class | Squared-exponential kernel, `kl.RBF(variance=..., lengthscale=...)`. |
| `to_operator` | function | K(X, X) (+ noise · I) as a symmetric, PSD-tagged lineax operator. `kl.to_operator(kernel, X, noise=None)`. |
