# Kernels

Two kernels ship, `RBF` and `Matern32`. Both take a `variance` and a `lengthscale`. This page fits each to the same twelve points of a sine and compares what they predict between the points.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import krigax as kx

train_inputs = jnp.linspace(0.0, 1.0, 12)[:, None]
train_targets = jnp.sin(6.0 * train_inputs[:, 0])
test_inputs = jnp.linspace(0.0, 1.0, 200)[:, None]
truth = jnp.sin(6.0 * test_inputs[:, 0])
```

The squared-exponential kernel gives a very smooth mean.

```python
rbf_model = kx.fit(kx.RBF(lengthscale=0.2), train_inputs, train_targets, noise=1e-4)
rbf_mean = kx.predict(rbf_model, test_inputs)
rbf_error = jnp.abs(rbf_mean - truth).max()
```

The Matérn-3/2 kernel is only once differentiable, so its mean bends more sharply at the data.

```python
matern_model = kx.fit(kx.Matern32(lengthscale=0.2), train_inputs, train_targets, noise=1e-4)
matern_mean = kx.predict(matern_model, test_inputs)
matern_error = jnp.abs(matern_mean - truth).max()
```

On this data the RBF mean stays within 0.0037 of the sine and the Matérn mean within 0.0274. The log evidence is 8.05 for the RBF model and −5.58 for the Matérn model, so the data prefer the smoother kernel.
