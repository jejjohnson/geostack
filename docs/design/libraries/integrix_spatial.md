---
title: "integrix: integrals over space"
short_title: "integrix over space"
description: "What integrix needs to integrate over areas, cells and lines as well as it integrates over time: the weights of a grid rule, masks and NaNs, weight functions with their own batch axes, and line integrals and fluxes. Four decisions, all decided on 10 Nov 2026 as recommended."
date: 2026-11-10
keywords:
  - integrix
  - design
  - spatial integrals
  - GeoStack
---

integrix computes one integral, $I = \int_\Omega f \mathrm{d}\mu$, as a rule, a scan or a march, and Waves 0 to 4 built all three. The steppers and their adjoints took the most work, so the library reads as a time library, but only `ix.march` and `ix.adjoint` are about time: `ix.rule`, `ix.scan`, `ix.adapt` and `ix.expect` integrate over whatever axis they are given. This note asks what is still missing when that axis is space: an area, a set of cells, or a line across a plume. It records four decisions (10 to 13, after the nine in the [integrix design](integrix.md)). The maintainer settled all four on 10 Nov 2026, each as recommended ([integrix#47](https://github.com/jejjohnson/integrix/issues/47)).

## Summary

| # | Decision | Recommended | Alternative |
| --- | --- | --- | --- |
| 10 | The shape of a multi-dimensional rule's weights | functax's grid rules keep the grid's shape: cell areas on an $N_y \times N_x$ grid are an $N_y \times N_x$ array | flattened weights, and `ix.rule` reshapes the values |
| 11 | Masks and NaNs | `ix.rule.integrate` and `ix.rule.weighted_mean` take `where=` and `nan_policy="propagate" \| "omit"` | callers fold masks into the weights and zero their NaNs |
| 12 | A weight function with its own batch axes | `weighted_mean`'s `weight` may carry leading axes, one result per leading index | the caller `vmap`s `weighted_mean` |
| 13 | Line integrals and fluxes | functax owns rules on curves (nodes, arc-length weights, normals); integrix contracts them and adds no line primitive | an `ix.rule.flux(curve, field, velocity)` that samples the field itself |

## Which integrals are spatial, and how they differ

The axes differ in ways that matter to the code, even though each case is one integral.

| Axis | What makes it different | integrix today |
| --- | --- | --- |
| Time | causal: the integrand depends on the solution, so it is stepped, with stability limits and adjoints that store the past | `ix.march`, `ix.adjoint` |
| A vertical column | one dimension, samples on given levels, often cumulative (optical depth) | `ix.rule`, `ix.scan` |
| An area or a volume | two or three dimensions, cells of unequal area (projected or on the sphere), masks, NaN pixels | `ix.rule` with a weights array; the weights are the caller's until functax's cell-area rules land |
| A line across a field | a flux through a transect, $\int c \mathrm{d}\ell$ or $\int c (\vect{u} \cdot \vect{n}) \mathrm{d}\ell$ | nothing |
| A spectral band | a response function weights the band; lines can be narrow and peaked | `ix.rule.weighted_mean`, `ix.adapt.GaussKronrod` |
| Directions | Gauss rules in $\mu = \cos \theta$, or nodes on the sphere | the rules are functax's and gaussx's; integrix only sums (no caller yet) |
| A probability space | a measure to sample from | `ix.expect`, and gaussx for Gaussians |

Time and probability are covered. The spatial rows are where the stack's code does things `ix.rule` cannot yet express.

## Running example: a methane plume on a 20 m grid

**TL;DR.** From one image of a methane plume we want two numbers: the mass of methane in the plume, and the flux of methane through lines drawn across it downwind. Both are spatial integrals, one over an area and one along a line, and both must skip cloudy pixels.

**Problem.** An image on a projected grid has $N_y \times N_x$ pixels of side $w = 20$ m and area $\Delta A = w^2$. In each pixel $i$, $\Omega_i$ is the methane column enhancement in kg m⁻², NaN where the pixel is cloudy. A boolean plume mask $m_i$ marks the plume, and $\vect{u}$ is the wind vector in m s⁻¹ with speed $U = \lVert \vect{u} \rVert$. The integrated mass enhancement over the plume $P$ is

```{math}
\text{IME} = \int_P \Omega \mathrm{d}A \approx \sum_{i \in P} \Omega_i \Delta A,
```

and the cross-sectional flux through a transect $L$ across the wind, with $\vect{n}$ its unit normal, is

```{math}
\Phi = \int_L \Omega (\vect{u} \cdot \vect{n}) \mathrm{d}\ell .
```

Today plumax computes both with NumPy at plumax `208717e`, in `src/plumax/geo/plume.py`:

- `IMEEstimate` (line 632): `ime_kg = float(np.nansum(enhancement[mask] * area))`, a masked sum that skips NaNs, with `area` from `pixel_area(gt.transform)`.
- `CrossSectionalFlux` (line 752): `flux = float(np.nansum(enhancement[on_transect] * wind_norm * width))`, where `on_transect` is the plume mask inside a strip one pixel wide across the wind, at a set distance downwind of the source. With $\vect{n}$ along the wind, $\vect{u} \cdot \vect{n} = U$.

Two other consumers integrate over space with different NaN rules:

- xrops' `column_integral` (`packages/xrtoolz-ops/src/xrops/atm/_src/vertical.py` @ xrtoolz `b3855d7`) is a trapezoid rule over levels whose docstring says a NaN at any level makes the whole column NaN.
- radtrax's SRF band integrals (`src/radtrax/instrument/srf.py` @ radtrax `0aff245`) weight one radiance spectrum by $B$ band responses $R_b(\lambda)$, a $B \times G$ array on $G$ wavelengths.

So the same contraction is wanted with NaNs skipped (plumax) and with NaNs propagated (xrops), over a masked area, along a line, and against a whole stack of response functions at once.

## What integrix offers today

`ix.rule.integrate(rule, values, *, axis=None)` contracts the values with a rule's weights over one axis or several, batched over the rest, and returns an `Integral`. The rule is anything with `weights` (functax's `Rule`), a tuple of 1-D rules (a tensor rule), or a plain weights array of any rank, whose shape must equal the values' shape along the integrated axes. `ix.rule.weighted_mean(rule, values, *, weight, axis=None)` divides $\sum w R f$ by $\sum w R$, with $R$ one value per node or per sample. Sums are pairwise, and weights take the values' dtype.

So `ix.rule.integrate(jnp.full((ny, nx), area), enhancement)` already computes an unmasked IME. What it cannot do is skip the cloudy pixels, keep the gradient finite when a NaN pixel is masked out, take $B$ response functions in one call, or integrate along a line.

## Decision 10: the weights of a grid rule keep the grid's shape

**Question.** When functax returns a rule for a two-dimensional grid (cell areas, functax#59), are its weights an $N_y \times N_x$ array or a flat vector of $N_y N_x$?

**Why now.** functax#59 sketches `cell_area(grid) -> Rule` with nodes of shape $(N_y N_x, 2)$ and weights of shape $(N_y N_x,)$. `ix.rule.integrate` matches the weights' shape to the values' integrated axes, so those weights cannot meet an $N_y \times N_x$ image without a reshape. integrix's own design sketch, `ix.rule.integrate(pixel_rule, enhancement)` with `pixel_rule = fx.quad.cell_area(plume_grid)` of shape $(N_y, N_x)$, assumes the other layout.

### A. Grid-shaped weights (recommended)

```python
# Shapes: Ny × Nx cells

# proposed (functax)
pixel_rule = fx.quad.cell_area(plume_grid)  # weights (Ny, Nx), nodes (Ny, Nx, 2)
# proposed (integrix), already what ix.rule.integrate does with a weights array
# IME = ∑ᵢ Ωᵢ ΔAᵢ
ime = ix.rule.integrate(pixel_rule, enhancement).value  # (Ny, Nx) → ()
```

### B. Flattened weights

```python
# proposed (functax), as functax#59 sketches it
pixel_rule = fx.quad.cell_area(plume_grid)  # weights (Ny·Nx,), nodes (Ny·Nx, 2)
# proposed (integrix): a new rule shape that flattens the values to match
ime = ix.rule.integrate(pixel_rule, enhancement, axis=(-2, -1)).value
```

| Criterion | A | B |
| --- | --- | --- |
| Works with `ix.rule` today | yes | no: needs a flatten rule in integrix |
| Matches a raster's layout (xarray, `GeoTensor`) | yes | no |
| Scattered points (no grid) | the same rule with one node axis | the same |
| Spherical cells | yes, weights in m² on a sphere of radius $a$ | yes |

**Decided (10 Nov 2026): A**, as recommended. The rule's `weights` keep the grid's shape and its `nodes` gain a trailing coordinate axis. A rule over scattered points keeps one node axis, as now. Alternative: B, if functax wants every rule to have exactly one node axis; integrix would then flatten the values' integrated axes in row-major order.

## Decision 11: masks and NaNs, with `where=` and `nan_policy=`

**Question.** How does a caller integrate over a region (a plume mask, a coastline) and decide what a NaN sample means?

**The trap.** Folding a mask into the weights, $\sum_i (w_i m_i) f_i$, does not skip NaNs: $0 \times \text{NaN}$ is NaN. Zeroing the NaNs first with `jnp.where(jnp.isnan(f), 0, f)` fixes the value but not always the gradient. If $f$ came from a computation that produced the NaN, the untaken branch of the `where` still sends NaN back through reverse mode. A safe version needs the guarded double `where` from integrix's CODE_REVIEW, and every caller would rewrite it.

### A. `where=` and `nan_policy=` on the contraction (recommended)

```python
# Shapes: Ny × Nx pixels

# proposed (integrix)
# IME = ∑ᵢ mᵢ Ωᵢ ΔAᵢ, NaN pixels skipped (np.nansum's meaning)
ime = ix.rule.integrate(pixel_rule, enhancement, where=plume_mask, nan_policy="omit").value  # () kg
# proposed (integrix): xrops keeps its NaN-propagating column
total = ix.rule.integrate(level_rule, density, axis=-1, nan_policy="propagate").value  # (…, Nz) → (…)
```

- `where` is a boolean array that broadcasts against the values; a sample with `where=False` contributes nothing, and its gradient is zero.
- `nan_policy="propagate"` (the default) is today's behaviour: any NaN among the selected samples makes the result NaN.
- `nan_policy="omit"` skips NaN samples, as `np.nansum` does, with a zero gradient for them.
- Both are applied with guarded `where`s, so no NaN reaches the gradient of the samples that count.
- `weighted_mean` renormalises over what is kept, $\sum_{i \in K} w_i R_i f_i / \sum_{i \in K} w_i R_i$, with $K$ the selected, non-NaN samples.
- The `Integral` gains no field; whether to report the count of samples kept is open.

### B. The caller folds the mask into the weights

```python
# today (plumax), as a JAX call: the caller guards the NaNs itself
safe = jnp.where(plume_mask & ~jnp.isnan(enhancement), enhancement, 0.0)
ime = ix.rule.integrate(pixel_rule, safe).value
```

| Criterion | A | B |
| --- | --- | --- |
| NaN-safe gradient | yes, guarded once in integrix | only when each caller guards correctly |
| Matches `np.nansum` and xrops' NaN rule | both, by `nan_policy` | each caller re-derives it |
| `weighted_mean` renormalisation | built in | the caller divides by a second masked sum |
| `jit` / `vmap` | yes: static shapes, the mask is data | yes |

**Decided (10 Nov 2026): A**, as recommended, on `ix.rule.integrate` and `ix.rule.weighted_mean`; it shipped in integrix 0.0.4. Alternative: B, with a documented recipe in integrix's guide. `ix.scan.cumulative` is left out: a running integral with skipped samples has no single obvious meaning, so a caller who needs it can ask.

## Decision 12: a weight function with its own batch axes

**Question.** Can `weighted_mean` take $B$ response functions at once, $\bar L_b = \sum_k w_k R_{bk} L_k / \sum_k w_k R_{bk}$?

**Why.** radtrax's `SpectralResponseFunction` stores a $B \times G$ response matrix (`srf.py` @ `0aff245`), and radtrax#307 leaves open whether `weight` may carry the band axis. Today `weight` must have the rule's shape or the values' shape, so a caller writes `jax.vmap(lambda r: ix.rule.weighted_mean(rule, radiance, weight=r))(response)`.

### A. Leading axes on `weight` (recommended)

```python
# Shapes: B bands; G wavelengths; P pixels

# proposed (integrix)
# L̄_b = ∑ₖ wₖ R_bk Lₖ / ∑ₖ wₖ R_bk
bands = ix.rule.weighted_mean(rule, radiance, weight=response)  # (P, G), (B, G) → (P, B)
```

The leading axes of `weight` beyond the rule's shape are batch axes of the weight function. The result has the values' batch axes first, then the weight's: $(P, B)$, a spectrum of band values per pixel.

### B. `vmap` at the call site (today)

```python
# today (integrix): a vmap over bands
bands = jax.vmap(lambda r: ix.rule.weighted_mean(rule, radiance, weight=r), out_axes=-1)(response)  # (P, B)
```

| Criterion | A | B |
| --- | --- | --- |
| One call per instrument | yes | no |
| Ambiguity with the per-sample weight | resolved by shape: a per-sample weight has the values' shape | none |
| Cost | one contraction, $O(PBG)$ | the same under `vmap` |

**Decided (10 Nov 2026): A**, as recommended, in `ix.rule.weighted_mean` only; it shipped in integrix 0.0.4. Alternative: B, with `vmap` documented in the guide; it already works.

## Decision 13: line integrals and fluxes

**Question.** Where do line integrals live, and what does integrix add for them?

A line integral has two discretisations, and the stack uses one of them today.

- **The strip form.** $\int_L \Omega \mathrm{d}\ell \approx \frac{1}{w} \sum_{i \in S} \Omega_i \Delta A_i$, the masked area integral over the pixels $S$ whose centres fall in a strip of width $w$ across the line. This is what plumax's `CrossSectionalFlux` computes, with $\Delta A = w^2$. It needs nothing beyond Decision 11.
- **The sampled form.** $\int_L c \mathrm{d}\ell \approx \sum_k c(\vect{x}_k) \Delta s_k$, with nodes $\vect{x}_k$ along the curve and arc-length weights $\Delta s_k$. The field is interpolated to the nodes. It handles a curved or slanted transect without staircase pixels, and a flux uses $c(\vect{x}_k) (\vect{u}(\vect{x}_k) \cdot \vect{n}_k)$.

### A. Rules on curves in functax; integrix contracts them (recommended)

```python
# Shapes: K nodes on the transect; Ny × Nx grid

# proposed (functax): a rule on a polyline, with the unit normal at each node
transect = fx.quad.polyline(vertices, n=K)  # nodes (K, 2), weights Δs (K,), normals (K, 2)
# today (functax): sample the gridded fields at the nodes
plan = fx.interp.Plan(...)  # the field read-out H at transect.nodes
# proposed (integrix): Φ = ∑ₖ Δsₖ Ωₖ (uₖ · nₖ)
normal_wind = einx.dot("k d, k d -> k", wind_at_nodes, transect.normals)
flux = ix.rule.integrate(transect, plan(enhancement) * normal_wind).value  # () kg s⁻¹
# proposed (integrix): the strip form, as plumax computes it today
flux_strip = ix.rule.integrate(pixel_rule, enhancement, where=in_strip & plume_mask, nan_policy="omit").value * wind_speed / w
```

### B. `ix.rule.flux` in integrix

```python
# proposed (B): integrix samples the field and forms u · n itself
flux = ix.rule.flux(transect, enhancement, wind, grid=plume_grid)
```

| Criterion | A | B |
| --- | --- | --- |
| Ownership (decision 3: rules are functax's) | kept: the curve rule is data | integrix would choose an interpolation scheme |
| New integrix API | none | one function, with grid and interpolation options |
| Reuse by tracerax and radiomax (line-of-sight and pixel-footprint integrals) | the same curve rule | a second entry point |
| plumax's strip form | `where=` from Decision 11 | still needs `where=` |

**Decided (10 Nov 2026): A**, as recommended. functax adds rules on curves, a `Rule` whose nodes lie on a polyline and whose weights are arc lengths, with the unit normals alongside (functax decides whether the normals are a field of the rule or a separate return). integrix adds nothing line-specific, and its guide shows both forms. Alternative: B, if the same three-step composition turns up in several consumers.

## Out of scope

- **Adaptive cubature in two or more dimensions.** quadax has one, but no consumer integrates a two-dimensional function to a tolerance; `ix.adapt` stays one-dimensional until one does.
- **Conservative remapping between grids.** It is an integral of one grid's field over another grid's cells, and it belongs to functax (functax#40, functax#41).
- **Spectral transforms** (Fourier, Chebyshev, spherical harmonics) are functax's and spectraldiffx's.
- **Masked scans** (`ix.scan.cumulative` with `where=`), until a caller needs one.

## Gates

| Gate | Passes when |
| --- | --- |
| G6-mask | with `nan_policy="omit"` and `where=`, `ix.rule.integrate` equals `np.nansum(f[mask] * w[mask])` to 1e-14 relative in float64, on random fields with NaNs; with `"propagate"`, a NaN among the selected samples gives NaN |
| G6-grad | gradients with respect to the kept samples and the weights match central differences to 1e-6, and are finite and exactly zero for masked and NaN samples |
| G6-batch | `weighted_mean` with a $(B, G)$ weight equals the `vmap` over bands to 1e-14 |
| G6-line | on a Gaussian plume $\Omega(x, y) = \frac{Q}{U \sqrt{2 \pi} \sigma} e^{-y^2 / 2 \sigma^2}$, the sampled form of $\int \Omega U \mathrm{d}y$ across the wind recovers $Q$ to the rule's discretisation error, and the strip form equals plumax's `CrossSectionalFlux` at `208717e` to 1e-12 |
| G6-port | plumax's `IMEEstimate` and `CrossSectionalFlux` on `ix.rule` reproduce their results at `208717e` to 1e-12, on the repository's own test fixtures |

## Roadmap

These become Wave 6 of integrix, an epic in integrix under [geostack#136](https://github.com/jejjohnson/geostack/issues/136), with one issue per decision that needs code:

- `[Design]` spatial integrals: grid-rule weights, masks and NaNs, batched weights and line integrals (this note's four decisions, settled in integrix#47 on 10 Nov 2026);
- `feat(rule)`: `where=` and `nan_policy=` on `integrate` and `weighted_mean` (Decision 11);
- `feat(rule)`: weight functions with their own batch axes in `weighted_mean` (Decision 12);
- `docs(guide)`: spatial integrals, with the IME, the cross-sectional flux in both forms, and a NaN-propagating column (Decisions 11 and 13);
- in functax, `feat(quad)`: rules on curves, with arc-length weights and unit normals (Decision 13), and the grid-shaped weights of functax#59 if Decision 10 goes to A;
- in plumax, a consumer issue: `IMEEstimate` and `CrossSectionalFlux` on `ix.rule` with masks (gate G6-port).

## Sources

- Code read on 10 Nov 2026: plumax `src/plumax/geo/plume.py` @ `208717e` (`IMEEstimate`, `CrossSectionalFlux`); xrops `packages/xrtoolz-ops/src/xrops/atm/_src/vertical.py` @ xrtoolz `b3855d7` (`column_integral`); radtrax `src/radtrax/instrument/srf.py` @ `0aff245`; integrix `src/integrix/_src/rule/_rule.py` @ v0.0.3.
- Issues: functax#44 (1-D rules), functax#45 (sphere rules) and functax#59 (cell-area rules), all open on 10 Nov 2026; radtrax#307 (the open point on a band axis); plumax#247 (which leaves the IME to its owner).
