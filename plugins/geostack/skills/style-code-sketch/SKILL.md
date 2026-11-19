---
name: style-code-sketch
description: Eman's pseudocode standard for READMEs, docs pages, design pages, reports and GitHub issues: a shapes header, then above each line a status marker, the equation and the shape. Use for any pseudocode about his stack.
---

# Code-sketch standard

Draft 2 · 19 Nov 2026 · the shape comment moves above the line, and what `style-python` already says is no longer repeated here.

A sketch reads as maths first and Python second: above each operation sit its status, its equation and its shape, and the code comes last.

A sketch is Python, so everything in `style-python` holds: the imports, einx and lineax, typed signatures, descriptive names, one operation per line, the equation above each mathematical line, comments above the line and never beside it, 88 columns.
The symbols and the Unicode in the comments follow `style-math`.
This skill adds only what a sketch needs beyond real code: a sketch states its sizes, shows the shape of each result, and may call names that do not exist yet.

## 1. Shapes header

The block opens with one comment naming every dimension and its size, if known:

```python
# Shapes: N_Ωx = 1,600 grid cells; N_Ωy observed cells; N_Sy = 12 channels
```

A dimension keeps its name in every block of a page; rename on collision.
Header names are the maths sizes of `style-math` (`references/indices.md`), and each has a matching jaxtyping dim in the table of `style-python` §3.

## 2. Above each line: marker, equation, shape

Each operation carries up to three comment lines above it, in this order:

```python
# today (kernellib)
# Σ_x = K_SS = [k(sᵢ, sⱼ; θ)]ᵢⱼ
# (N_Ωx, 3) → (N_Ωx, N_Ωx) operator
prior_cov = kl.to_operator(kernel.frozen(), cell_xyz)
```

- **The marker** (§3) is there on every call into the stack.
- **The equation** is there on every mathematical line, as in `style-python` §7. In a sketch every symbol in it is defined in the prose before the block, or in the shapes header.
- **The shape** is there on every bare assignment of an array: the transition `# (N_Ωx, 2) → (N_Ωx, 3)`, or `# → (N_Ωx,)` when the inputs are plain from the lines above. Scalars are `# → ()`; distributions give their event shape, `# event (T, N_Ωy)`; operators say so, `# (N_Ωx, N_Ωx) PSD operator`.

Real code has no shape comments, because its types carry the shapes (`style-python` §3). A sketch shows intermediate results that have no signature, so it writes their shapes, and it writes them above the line like every other comment.

**Signatures** are typed as in `style-python` §3. Carriers use the same form, and the string keeps the library's own dim names:

| Annotation | Means |
| --- | --- |
| `Float[Array, "obs channel"]` | a raw array |
| `GeoArray["patch band y x"]` | a geoarray carrier |
| `GeoSwath["band row col"]`, `GeoPoints["time variable obs"]` | swath and point carriers |
| `DataArray["time lat lon"]`, `Dataset["time lat lon"]` | xarray carriers |

Repeated long types get a short alias, defined once: `Op = lx.AbstractLinearOperator`.

## 3. Status markers

Every call into the stack has a marker on its own line, first of the comment lines above it:

| Marker | Means |
| --- | --- |
| `# today (pkg)` | the name ships at the pinned version (the capability index, where one exists) |
| `# today* (pkg)` | the name ships, but the arguments are sketched |
| `# proposed (X)` | not shipped; X is the workstream, proposal or issue that adds it |
| `# user` | a function the page or its notebook defines, not a stack name |
| `# not pinned (pkg)` | lives in a library whose API was not checked for this page |

Never mark a name `today` without checking it in the repo.
A marker names the package and not its version; the page's pin list says which version was checked.

## 4. Target-state pages

A design page that shows the end state, with every finding applied, tags each line that depends on a finding, in place of the marker:

```python
# ← CS-F3
alpha: GeoArray["patch y x"] = enhancement(scene)
```

The gap under the block says what is missing today.
Use this form instead of status markers only when the page's subject is the target state.

## 5. Refactors: before and after

A refactor shows both versions in one block, current code first:

```python
# Today:
tapered = taper_matrix * covariance
# After:
# proposed (A)
tapered = kl.TaperLocalizer(taper)(covariance)
```

## 6. Verification

- Every `today` block is run as written before it ships; README blocks always are.
- `proposed` blocks are not run, but their `today` calls are checked by name.
- Maths in the surrounding Markdown passes `check_math.py` from `style-math`.

## Conforming sketch

Prose before the block: μ_x is the prior mean, Σ_x = K_SS the prior covariance built from the kernel k(·, ·; θ) on the cell coordinates S, Σ_ε the observation-error covariance, and H the 0/1 matrix that selects the observed cells.

```python
import geonnax as gnx
import kernellib as kl
import vardax as vdx

# Shapes: N_Ωx = 1,600 grid cells; N_Ωy observed cells

# today (geonnax)
# s(λ, φ) = (cos φ cos λ, cos φ sin λ, sin φ) ∈ S²
# (N_Ωx, 2) → (N_Ωx, 3)
cell_xyz = gnx.geo.lonlat_to_cartesian3d(cell_lonlat)

# today (kernellib)
# Σ_x = K_SS = [k(sᵢ, sⱼ; θ)]ᵢⱼ
# (N_Ωx, 3) → (N_Ωx, N_Ωx) operator
prior_cov = kl.to_operator(kernel.frozen(), cell_xyz)

# proposed (C)
# log p(y; θ) = log 𝒩(y; H μ_x, H Σ_x Hᵀ + Σ_ε)
# → ()
log_evidence = vdx.oi_marginal(
    background_mean, prior_cov, obs_cov, mask=observed
)
```

The names here come from a design guide of 7 Oct 2026; re-check them before reusing this block.

## Review checklist

- [ ] The block passes the `style-python` checklist where it applies to a sketch (einx and lineax, one operation per line, descriptive names, 88 columns).
- [ ] Shapes header first; names consistent across the page.
- [ ] Above each line, in order: the marker on a stack call, the equation on a mathematical line, the shape on a bare array assignment. No comment beside a line.
- [ ] Every symbol defined before the block or in the header.
- [ ] Every `today` name checked in the repo; `today` blocks run as written.
- [ ] Refactors shown as `# Today:` / `# After:`.
