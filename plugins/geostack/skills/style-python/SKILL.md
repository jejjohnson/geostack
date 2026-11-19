---
name: style-python
description: Eman's Python house style for JAX/Equinox scientific code — docstrings, typing, module design, naming, errors, tests. Use when writing, reviewing or refactoring Python in his repos.
---

# Python house style

Draft 5 · 19 Nov 2026 · the open questions are settled, and what `style-math` and `style-code-sketch` own is pointed to instead of repeated.

Three skills share the ground, and each rule lives in one of them. `style-math` owns the symbols, their roles, the sizes and how maths is written in Unicode. This skill owns how they meet real code: the dim names, the variable names, the comments, the modules. `style-code-sketch` builds on this one and owns what only pseudocode needs: the shapes header, the shape comment and the status markers.

Code reads as maths first and Python second.
Each line does one thing, the equation it computes sits above it, and the types carry the shapes.

## 1. Layout

- **`_src` package.** All code lives in `pkg/_src/`. Public modules only re-export (`from ._src.solvers import BeerLambert`). Nothing outside `_src` defines logic.
- **One concept per file.** A protocol and its implementations share a file (`solvers.py` holds `RTSolver` and `BeerLambert`). Split when a second concept appears, not at a line count.
- **Optional dependencies behind extras,** imported lazily inside the function that needs them.
- **Every module opens with a docstring** saying what it holds, in one or two sentences.

## 2. Imports

Modules under fixed aliases, never bare function imports:

```python
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, PRNGKeyArray
```

`jaxtyping` names are the one exception, since they appear in every signature.

## 3. Types and shapes

- Every array argument and return is annotated with jaxtyping and named dims: `Float[Array, "obs channel"]`.
- Shapes live in types, never in comments. A sketch is the one place that writes a shape in a comment, above the line (`style-code-sketch` §2).
- Dim names come from one vocabulary, shared with xarray dims. Each stands for one maths size, and the sizes themselves are defined in `style-math` (`references/indices.md`):

| Dim | Maths size | Meaning |
| --- | --- | --- |
| `sample` | $`N = \vert \mathcal{D} \vert`$ | independent data pairs |
| `test` | $`M`$ | test inputs |
| `member` | $`N_{\mathcal{E}}`$ | ensemble members, Monte Carlo draws |
| `batch`, `patch`, `tile` | $`N_\text{batch}`$, $`N_\text{patch}`$, $`N_\text{tile}`$ | pieces of a dataset or domain |
| `time` | $`N_{\mathcal{T}_x}`$ | model time steps |
| `cell` | $`N_{\Omega_x}`$ | grid cells where the state lives |
| `obs` | $`N_{\Omega_y}`$, or $`D_y`$ | observation points (pixels, stations); the whole observation vector when it is flat and not split into points and channels |
| `channel` | $`N_{\mathcal{S}_y}`$ | spectral channels |
| `var`, `level` | $`N_{\mathcal{V}_x}`$, $`N_{\mathcal{P}_x}`$ | physical variables, vertical levels |
| `inducing` | $`N_{\mathcal{I}}`$ | inducing points |
| `state`, `latent`, `param` | $`D_x`$, $`D_z`$, $`D_\theta`$ | lengths of flattened vectors |

  Library carriers keep their own dim strings (`GeoArray["patch band y x"]`; the forms are in `style-code-sketch` §2); those names stay inside the type and never reach an equation, where space is $`\mathbf{s}`$.
- Types are checked at runtime in tests only, through the import hook in `conftest.py`:

```python
from jaxtyping import install_import_hook

with install_import_hook("radtrax", "beartype.beartype"):
    import radtrax
```

## 4. Array operations

- einx for reductions, contractions, broadcasts and rearranges. Never `axis=`, `.T`, `reshape` or `[:, None]`.
- einx works on dense arrays only. A linear operator (lineax, gaussx) keeps its own methods (`.mv`, `.T`, `sandwich`, the solve primitives), and is never materialised just to use einx.
- When a contraction has two axes of the same dim, name them by role (`next prev` for consecutive states, `row col` for the two sides of a matrix), never `i j`.
- A function over the repo's size limit is split into private helpers that take everything as arguments (`_kalman_predict`, `_kalman_update`). Docstrings and equation comments are never cut to fit; where a repo counts them, its limit should count code lines only.
- lineax for linear solves; solve, never invert.
- One operation per line. No nested calls that hide an intermediate result.
- Lines stay within ruff's 88 columns; wrap a long call over several lines rather than shortening names. A long einx call is the usual offender, in tests as much as in the library, so run `ruff check` and `ruff format` on every file before handing it over.

## 5. Modules and functions

- **Arrays live in `eqx.Module`s** and are always leaves, never static fields.
- **Host-side records that are never traced** (configs for offline tools, mutable NumPy accumulators) are plain frozen dataclasses.
- **Modules carry state, functions carry transforms.**
- **Hyperparameters are fields of a callable module; data goes into `__call__`.** This is the semicolon of the maths: in $`f(\mathbf{x}, \mathbf{u}; \boldsymbol{\theta})`$ the fixed $`\boldsymbol{\theta}`$ are fields and the variables are arguments, `dynamics(state, inputs)`.
- **One parameterisation per module.** A module stores a covariance or a precision as a lineax operator, never both; the other is reached by a solve.
- **Inputs that travel together form their own small module** (μ and Σ become `Background`).
- **Protocols are abstract modules** with `eqx.AbstractVar`. In a repo whose modules open with `from __future__ import annotations` (plumax, and any repo whose `AGENTS.md` asks for it), `AbstractVar` is not seen: the attribute becomes an ordinary field and a subclass that gives it a default breaks the field order. There, declare it on the base as an abstract property (`@property` over `@abc.abstractmethod`) and let each subclass provide it as a field.
- **Many inputs:** few positionals, the rest keyword-only after `*`.
- **PRNG keys** are keyword-only, named `key`, last. Split once at the top into named keys, one per stochastic stage.
- **Updates.** Public `with_*` methods. Config fields use `dataclasses.replace` so `__check_init__` re-runs; array leaves use `eqx.tree_at`.
- **Transforms:** `vmap` sits outside `jacrev`; a swath is `filter_vmap(jacrev(...))`.

## 6. Naming

- Descriptive names everywhere: `posterior_precision`, not `Lam`.
- The symbol appears only in the equation comment above the line, so that comment is mandatory on every mathematical line.
- Names come from the symbol's role and subscript, so each name points back to one symbol. The symbols and their roles are `style-math` §3 and §4; this table gives each its name in code, and a symbol added there gets its name here:

| Symbol | Name | Symbol | Name |
| --- | --- | --- | --- |
| $`\mathbf{x}`$, $`\mathbf{y}`$, $`\mathbf{z}`$, $`\mathbf{u}`$ | `state`, `obs`, `latent`, `inputs` | $`\boldsymbol{\theta}`$, $`\boldsymbol{\nu}`$ | `params`, `variational_params` |
| $`\boldsymbol{\mu}_x`$, $`\boldsymbol{\Sigma}_x`$, $`\boldsymbol{\Lambda}_x`$ | `prior_mean`, `prior_covariance`, `prior_precision` | $`\boldsymbol{\Sigma}_\varepsilon`$, $`\boldsymbol{\Lambda}_\varepsilon`$ | `obs_noise_covariance`, `obs_noise_precision` |
| $`\boldsymbol{\mu}_y`$, $`\boldsymbol{\Sigma}_y`$ | `obs_mean`, `obs_covariance` | $`\boldsymbol{\Sigma}_{xy}`$ | `cross_covariance` |
| $`\boldsymbol{\mu}_{x \mid y}`$, $`\boldsymbol{\Lambda}_{x \mid y}`$ | `posterior_mean`, `posterior_precision` | $`\mathbf{G}`$, $`\mathbf{H}`$ | `gain`, `obs_operator` |
| $`\mathbf{J}_h`$, $`\mathbf{J}_f`$ | `obs_jacobian`, `dynamics_jacobian` | $`\mathbf{K}_{XX}`$, $`\mathbf{K}_{*X}`$ | `gram_train`, `gram_test_train` |
| $`\mathcal{J}`$, $`\mathcal{L}`$ | `cost`, `loss` | $`\mathbf{x}^{(e)}`$, $`\boldsymbol{\theta}^{(i)}`$ | loop over `member`, `iteration` |
| $`\boldsymbol{\Sigma}_\eta`$ | `model_noise_covariance` | $`\mathbf{r}_t`$ | `residual` |
| $`\mathbf{G}_{\text{smooth},t}`$ | `smoother_gain` | $`D_y`$, $`T`$ | `obs_size`, `num_steps` |

## 7. Comments

- Comments go above the line, never beside it.
- A unicode equation above each mathematical operation, in the notation of `style-math`: `# Λ_x|y = Hᵀ Λ_ε H + Λ_x`. Which characters to use is in that skill's `references/targets.md`.
- Step headers split a long function into stages: `# --- whiten ---`.
- Elsewhere, comments say *why*, never *what*.

## 8. Docstrings

Google layout, types left to jaxtyping:

- One-line summary, then the defining equation in unicode.
- `Algorithm:` with numbered steps for any function that implements a method. The reasons behind a step go in a comment at the line, not in the docstring.
- `Args:` gives each name its meaning and its `style-math` symbol, with no type.
- `Returns:` says what comes back, not its type. Skip it for a callable module whose `__call__` signature already says it.
- `Raises:` lists the package errors a caller can catch.
- `Examples:` gives 2–3 cases: typical, edge and composition. Each builds its own small data, so `pytest --doctest-modules` runs it as written. Outputs are shapes, `allclose` booleans or reprs, never raw floats. Error examples use `# doctest: +IGNORE_EXCEPTION_DETAIL`.
- `References:` when a paper defines the method.

## 9. Validation and errors

- Static checks on fields in `__check_init__`.
- Value checks on traced arrays with `eqx.error_if`, so they survive `jit`.
- One base class per package (`RadtraxError`); subclasses are named by what failed (`TableRangeError`, `ConvergenceError`, `GeometryError`).
- Each error also inherits the closest standard-library exception (`TableRangeError(RadtraxError, ValueError)`), so a caller who knows nothing of the package still catches it. integrix does this for every error.

## 10. Precision

The caller decides. Functions preserve the input dtype and never cast silently.
Constants are created in the input's dtype (`jnp.asarray(c, dtype=x.dtype)`) or as Python scalars.
Tests that need float64 enable x64 themselves.

## 11. Configuration

Objects round-trip through `get_config()` / `from_config()`; arrays are excluded.
Applications build configs with hydra-zen on top.

## 12. Scripts and CLIs

`cyclopts` for arguments and `loguru` for output. Never argparse, click or typer, and never `print` for progress or stdlib `logging`.

## 13. DRY

Extract only real duplicates: code that must change together.
Code that merely looks alike stays separate.

## 14. Tests

Every public forward function gets all four:

1. **Closed-form oracle** at a stated tolerance (`rtol=1e-12` in float64).
2. **Property-based** invariants with hypothesis.
3. **Gradient check:** `jacrev` against central differences.
4. **Transform invariance:** eager, `jit` and `vmap` agree with a Python loop.

Test code keeps the rules of §4 and §6: einx in place of `axis=`, `.T`, `reshape` and `[:, None]`, descriptive names, and the tolerance of each comparison stated. A property test that cannot use a drawn case rejects it with `hypothesis.assume`; a bare `return` counts it as a pass.

## Conforming example

Every example in its docstring passes doctest, and every line fits in 88 columns.

```python
# radtrax/_src/retrieval/matched_filter.py
"""Matched filter on arrays: background statistics and the filter itself."""

import dataclasses
from typing import Any

import einx
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from radtrax._src.errors import BackgroundError


class Background(eqx.Module):
    """Background statistics of a scene: mean μ_y and covariance Σ_y."""

    mean: Float[Array, "channel"]
    covariance: lx.AbstractLinearOperator


class MatchedFilter(eqx.Module):
    """Matched-filter enhancement for a known target signature.

    α̂(y) = J_hᵀ Σ_y,load⁻¹ (y − μ_y) / (J_hᵀ Σ_y,load⁻¹ J_h),
    with the loaded covariance Σ_y,load = Σ_y + β I.

    Algorithm:
        1. Load the covariance: Σ_y,load = Σ_y + β I.
        2. Solve for the filter direction w = Σ_y,load⁻¹ J_h.
        3. Project each residual y − μ_y onto w and normalise by J_hᵀ w.

    Args:
        ridge: Diagonal loading β ≥ 0 added to Σ_y before the solve.

    Raises:
        BackgroundError: If ``ridge`` is negative.

    Examples:
        Two pixels, three bands, white background:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(3))
        >>> background = Background(mean=jnp.zeros(3), covariance=covariance)
        >>> target = jnp.array([1.0, 0.5, 0.0])
        >>> radiance = jnp.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        >>> enhancement = MatchedFilter()(radiance, target, background)
        >>> enhancement.shape
        (2,)
        >>> bool(jnp.allclose(enhancement, jnp.array([2.0, 0.0])))
        True

        Change the loading, then round-trip the configuration:

        >>> loaded = MatchedFilter().with_ridge(0.5)
        >>> MatchedFilter.from_config(loaded.get_config())
        MatchedFilter(ridge=0.5)

        A negative ridge is rejected at construction:

        >>> MatchedFilter(ridge=-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        BackgroundError: ridge must be ≥ 0, got -1.0

    References:
        Thompson et al. (2015), AMT 8, 4383–4397.
    """

    ridge: float = 0.0

    def __check_init__(self) -> None:
        if self.ridge < 0.0:
            raise BackgroundError(f"ridge must be ≥ 0, got {self.ridge}")

    def __call__(
        self,
        radiance: Float[Array, "obs channel"],
        target: Float[Array, "channel"],
        background: Background,
    ) -> Float[Array, "obs"]:
        # NaNs in one pixel would poison the shared solve
        has_nan = jnp.isnan(radiance).any()
        radiance = eqx.error_if(radiance, has_nan, "radiance has NaN")

        # --- filter direction ---
        # Σ_y,load = Σ_y + β I
        structure = background.covariance.in_structure()
        identity = lx.IdentityLinearOperator(structure)
        loaded_covariance = background.covariance + self.ridge * identity
        # w = Σ_y,load⁻¹ J_h
        solution = lx.linear_solve(loaded_covariance, target)
        filter_direction = solution.value
        # J_hᵀ Σ_y,load⁻¹ J_h
        normalisation = einx.dot("channel, channel ->", target, filter_direction)

        # --- score ---
        # r = y − μ_y
        residual = einx.subtract(
            "obs channel, channel -> obs channel", radiance, background.mean
        )
        # rᵀ Σ_y,load⁻¹ J_h
        projection = einx.dot(
            "obs channel, channel -> obs", residual, filter_direction
        )
        # α̂ = rᵀ Σ_y,load⁻¹ J_h / (J_hᵀ Σ_y,load⁻¹ J_h)
        enhancement = projection / normalisation
        return enhancement

    def with_ridge(self, ridge: float) -> "MatchedFilter":
        # replace, not tree_at: re-runs __check_init__
        return dataclasses.replace(self, ridge=ridge)

    def get_config(self) -> dict[str, Any]:
        return {"ridge": self.ridge}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MatchedFilter":
        return cls(**config)
```

## Review checklist

A style review is still a review. A fault in what the code computes that is met on the way (a scaling applied twice, a division by zero for one member, state that is stored and never read) is reported first, ahead of every style point.

- [ ] Logic only under `_src`; public modules re-export.
- [ ] Every array typed with named dims; no shape comments.
- [ ] No `axis=`, `.T`, `reshape`, `[:, None]`; einx instead.
- [ ] One operation per line, equation comment above each mathematical one.
- [ ] Descriptive names from the symbol table; symbols only in comments.
- [ ] Dims from the shared vocabulary; symbols and Unicode follow `style-math`.
- [ ] Fixed parameters as module fields, variables as `__call__` arguments; one parameterisation per module.
- [ ] Module docstring present.
- [ ] Lines within 88 columns.
- [ ] Docstring: summary, equation, numbered algorithm, untyped Args, 2–3 doctested Examples.
- [ ] Field checks in `__check_init__`, array checks with `eqx.error_if`.
- [ ] Errors subclass the package base and the closest builtin, and name what failed.
- [ ] No silent dtype casts.
- [ ] `key` keyword-only and last.
- [ ] Oracle, property, gradient and transform tests for each public forward.

## Not covered yet

Control flow under `jit` (`jnp.where` against `lax.cond`), warnings, and public `__all__`. Notebooks follow each repository's own rule in its `AGENTS.md`, since the repositories differ on committing executed `.ipynb` files.
