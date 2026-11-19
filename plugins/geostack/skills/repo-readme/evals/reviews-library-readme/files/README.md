<div align="center">

<img alt="krigax" src="docs/assets/logo-light.svg" width="340">

**Fast Gaussian processes for JAX.**

[![Tests](https://github.com/jejjohnson/krigax/actions/workflows/tests.yml/badge.svg)](https://github.com/jejjohnson/krigax/actions/workflows/tests.yml)
[![Lint](https://github.com/jejjohnson/krigax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/krigax/actions/workflows/lint.yml)
[![Docs](https://github.com/jejjohnson/krigax/actions/workflows/docs.yml/badge.svg)](https://github.com/jejjohnson/krigax/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/jejjohnson/krigax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/krigax)

</div>

---

krigax is a library for Gaussian-process regression — it gives you kernels,
a fitted model and prediction, all in JAX. It was written because we needed
kriging in several projects and kept copying the same forty lines around.

```
+---------+      +--------+      +---------+
| kernel  | ---> |  fit   | ---> | predict |
+---------+      +--------+      +---------+
```

![overview](docs/assets/overview.png)

## 📥 Install

Python 3.10 or newer.

```bash
pip install krigax
```

## 🌟 Highlights

- ⚡ **Blazing fast**: scales to millions of points.
- 🧱 **Kernels**: `RBF` and `Matern32`.
- 🪶 **Sparse**: `SparseGP` with inducing points.
- 🔧 **Solvers**: swap `CholeskySolver` for `CGSolver`.

## 📐 The Core Idea: Why Kriging Works

Start from the joint Gaussian of the training targets and the test value,

$$\begin{bmatrix} \bm{y} \\ f_* \end{bmatrix} \sim \mathcal{N}\left(0, \begin{bmatrix} K_{XX} + \sigma^2 I & K_{X*} \\ K_{*X} & K_{**} \end{bmatrix}\right)$$

condition on $\bm{y}$,

$$p(f_* | \bm{y}) = \mathcal{N}(\mu_*, \sigma_*^2)$$

complete the square to get

$$\mu_* = K_{*X}(K_{XX} + \sigma^2 I)^{-1}\bm{y}$$

$$\sigma_*^2 = K_{**} - K_{*X}(K_{XX} + \sigma^2 I)^{-1}K_{X*}$$

and write the weights as

$$\alpha = (K_{XX} + \sigma^2 I)^{-1}\bm{y}$$

so that $\mu_* = K_{*X}\alpha$ where $K_{XX}$ is the kernel matrix and $\alpha_n$ is the weight of sample $n$.

## 🚀 Quickstart

```python
import jax.numpy as jnp
import krigax as kx

x = jnp.linspace(0, 1, 20)[:, None]  # (20, 1)
y = jnp.sin(6 * x[:, 0])  # (20,)
model = kx.fit(kx.RBF(lengthscale=0.2), x, y)
mu = model.kernel(x, x) @ model.weights  # (20,)
```

A second example, with Matérn and conjugate gradients:

```python
model = kx.fit(kx.Matern32(lengthscale=0.2), x, y, solver=kx.CGSolver())
```

And fitting the lengthscale with optax:

```python
import optax

opt = optax.adam(1e-2)
for step in range(500):
    grads = jax.grad(loss)(params)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)
```

## 🔬 What else is in the box

| Module | Symbols | Notes |
|---|---|---|
| `krigax` | `RBF`, `Matern32`, `fit`, `predict` | the basics |
| `krigax.sparse` | `SparseGP`, `variational_fit` | inducing points |

## 🧩 Ecosystem

| Concern | Owner |
|---|---|
| operators and solvers | lineax |

krigax is used by plumax, pyrox and filterax.

## 🎓 References

- Rasmussen and Williams (2006). [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/). *MIT Press*.
- Wilson and Nickisch (2015). Kernel Interpolation for Scalable Structured Gaussian Processes. *ICML*.
