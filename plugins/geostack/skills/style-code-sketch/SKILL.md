---
name: style-code-sketch
description: Eman's pseudocode standard for READMEs, design pages, reports and GitHub issues: shapes header, equation above each line, shape column, status markers. Use for any pseudocode about his stack.
---

# Code-sketch standard

Draft 1 · 24 Oct 2026 · merges three earlier versions of the standard.

A sketch reads as maths first and Python second: the equation sits above each operation, the shapes run down the right margin, and every call into the stack says whether it exists today.

Everything in `style-python` applies too (imports, einx, lineax, naming, one operation per line, 88 columns).
This skill adds what only sketches need, because a sketch may call names that do not exist yet.

## Sources

This standard replaces three versions, which did not fully agree:

1. "Design-plan structure & formatting guide" (artifact, 7 Oct 2026), exported as the GeoStack project's Report Formats doc: the shapes header, the equation above, the shape column and the first markers.
2. geostack `docs/overview_examples.md`, "Design sketches and status markers": the full marker vocabulary.
3. geostack `docs/design/case_studies.md`, "How to read the code": carrier annotations and finding tags.

Where they conflict, this file wins; fix the old page when you touch it.

## 1. Shapes header

The block opens with one comment naming every dimension and its size, if known:

```python
# Shapes: N_Ωx = 1,600 grid cells; N_Ωy observed cells; N_Sy = 12 channels
```

A dimension keeps its name in every block of a page; rename on collision.
Header names are the maths sizes of `style-math`: `D_x` for a vector's length, and a count named by its set (`N_Ωx` grid cells, `N_Ωy` observation points, `N_Sy` spectral channels, `N_E` ensemble members).
Each has a matching jaxtyping dim in `style-python`: `N_Ωx` is `cell`, `N_Ωy` is `obs`, `N_Sy` is `channel`, `N_E` is `member`.

## 2. Equation above each operation

A unicode equation sits on its own line above the line that computes it.
Every symbol is defined in the prose before the block, or in the shapes header.

```python
# Λ_x|y = Hᵀ Λ_ε H + Λ_x
posterior_precision = gauss_newton_term + prior_precision
```

## 3. Shapes: types on signatures, a column on assignments

- **Signatures** carry jaxtyping types with named dims, and carriers use the same style:

  | Annotation | Means |
  | --- | --- |
  | `Float[Array, "obs channel"]` | a raw array; dims from the `style-python` vocabulary |
  | `GeoArray["patch band y x"]` | a geoarray carrier; the string keeps the library's dim names |
  | `GeoSwath["band row col"]`, `GeoPoints["time variable obs"]` | swath and point carriers |
  | `DataArray["time lat lon"]`, `Dataset["time lat lon"]` | xarray carriers |

- **Bare assignments** end with the shape transition in one aligned column: `# (N_Ωx, 2) → (N_Ωx, 3)`.
  Scalars are `→ ()`; distributions give their event shape, `# event (T, N_Ωy)`; operators say so, `# (N_Ωx, N_Ωx) PSD operator`.
- This column is the one exception to "comments go above the line".
- A call wrapped over several lines puts its shape on the first line.
- Repeated long types get a short alias, defined once: `Op = lx.AbstractLinearOperator`.

## 4. Status markers

Every call into the stack has a marker on its own line, above the equation:

| Marker | Means |
| --- | --- |
| `# today (pkg)` | the name ships at the pinned version (the capability index, where one exists) |
| `# today* (pkg)` | the name ships, but the arguments are sketched |
| `# proposed (X)` | not shipped; X is the workstream, proposal or issue that adds it |
| `# user` | a function the page or its notebook defines, not a stack name |
| `# not pinned (pkg)` | lives in a library whose API was not checked for this page |

Never mark a name `today` without checking it in the repo.

## 5. Target-state pages

A design page that shows the end state, with every finding applied, tags each line that depends on a finding:

```python
alpha: GeoArray["patch y x"] = enhancement(scene)          # ← CS-F3
```

The gap under the block says what is missing today.
Use this form instead of status markers only when the page's subject is the target state.

## 6. Refactors: before and after

A refactor shows both versions in one block, current code first:

```python
# Today:
tapered = taper_matrix * covariance
# After:
# proposed (A)
tapered = kl.TaperLocalizer(taper)(covariance)
```

## 7. Verification

- Every `today` block is run as written before it ships; README blocks always are.
- `proposed` blocks are not run, but their `today` calls are checked by name.
- Symbols, shapes and Unicode in comments follow `style-math`; maths in the surrounding Markdown passes its `check_math.py`.

## Conforming sketch

Prose before the block: μ_x is the prior mean, Σ_x = K_SS the prior covariance built from the kernel k(·, ·; θ) on the cell coordinates S, Σ_ε the observation-error covariance, and H the 0/1 matrix that selects the observed cells.

```python
import geonnax as gnx
import kernellib as kl
import vardax as vdx

# Shapes: N_Ωx = 1,600 grid cells; N_Ωy observed cells

# today (geonnax)
# s(λ, φ) = (cos φ cos λ, cos φ sin λ, sin φ) ∈ S²
cell_xyz = gnx.geo.lonlat_to_cartesian3d(cell_lonlat)    # (N_Ωx, 2) → (N_Ωx, 3)

# today (kernellib)
# Σ_x = K_SS = [k(sᵢ, sⱼ; θ)]ᵢⱼ
prior_cov = kl.to_operator(kernel.frozen(), cell_xyz)    # (N_Ωx, 3) → (N_Ωx, N_Ωx) op

# proposed (C)
# log p(y; θ) = log 𝒩(y; H μ_x, H Σ_x Hᵀ + Σ_ε)
log_evidence = vdx.oi_marginal(                          # → ()
    background_mean, prior_cov, obs_cov, mask=observed
)
```

The names here come from the 7 Oct guide; re-check them before reusing this block.

## Review checklist

- [ ] Shapes header first; letters consistent across the page.
- [ ] Unicode equation above every mathematical line, in `style-math` notation; symbols defined before the block.
- [ ] Typed signatures; aligned shape column on bare assignments.
- [ ] A marker on every stack call; every `today` name checked in the repo.
- [ ] einx, not `axis=`, `.T`, `reshape` or `[:, None]`; lineax for solves.
- [ ] One operation per line, within 88 columns.
- [ ] Refactors shown as `# Today:` / `# After:`.
- [ ] `today` blocks run as written.

## Open questions

- Descriptive names (as in `style-python`) or the paper's symbols? Symbols make the shape column fit; this draft uses descriptive names.
- Should `today` always name its pin (`today (kernellib 0.0.18)`), or is the page's pin list enough?
