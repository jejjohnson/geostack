# Localized covariance and the kriging weights

The prior covariance is tapered so that distant cells do not interact, and the weights solve the regularised system.

```python
import jax.numpy as jnp
import kernellib as kl
import gaussx as gx

# today (kernellib 0.4.1)
K = kl.to_operator(kernel, X).as_matrix()  # (N, N)

# today (kernellib)
# Σ_loc = ρ ∘ Σ_x
K_loc = kl.TaperLocalizer(radius=3.0)(K)  # (N, N)

d = dist(X, X); w = jnp.exp(-d / 2.0)  # (N, N)

# (N,)
# α = (Σ_loc + λ I)⁻¹ y
a = jnp.linalg.inv(K_loc + lam * jnp.eye(N)) @ y
```

Old version of the prediction:

```python
pred = K_star @ a
```

New version of the prediction:

```python
pred = gx.solve(op, y)  # (n_cells,)
```
