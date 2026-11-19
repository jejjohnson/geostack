# Writing maths for each format

Write once with the macros in `macros.yml`; each format receives them differently.

| Format | Engine | Macros | Display maths | Labels |
| --- | --- | --- | --- | --- |
| GitHub (issues, PRs, `.md`) | MathJax 3, in the browser | none: run `expand_macros.py` first | a `math` code fence | none |
| MyST (journal, notebooks) | KaTeX, at build time | `math:` under `project:` in `myst.yml` | a `{math}` directive with `:label:`, or `$$` lines | `:label:`, or `(label)` after the closing `$$` |
| mkdocs | MathJax 3 or KaTeX, in the browser, via `pymdownx.arithmatex` (`generic: true`) | JavaScript config from `emit_macros.py` | `$$` lines | MathJax only |
| code comments, docstrings | none | none | Unicode, see below | none |

Rules that hold everywhere, because one source must render in all four:

- No spacing or sizing macros: `\,` `\;` `\:` `\!` `\quad` `\qquad` `\big` and family, `\tfrac`, `\dfrac`, `\boxed`, `\cdots`, `\operatorname`. Use plain spacing, `\frac`, `\ldots`, `\mathrm`. Split a display with several equations into several blocks instead of joining them with `\qquad`.
- No `\bm` (MathJax rejects it) and no `\mathbb{1}` (KaTeX draws a plain 1).
- Each `$…$` and `$$…$$` expression stays on one source line.

## GitHub

Markdown runs before MathJax, and the mobile app renders a subset.

| Do | Don't |
| --- | --- |
| Display maths in a `math` code fence | `$$` mid-line, or `$$` without a blank line before it |
| Inline `$…$` with no space just inside the dollars | `$ x $` |
| `` $`…`$ `` when the expression has `*` pairs, `\{ \}` or `\\` | `$a * b * c$`, `$\{x\}$` (Markdown eats the `*` and the backslashes) |
| `\vert`, `\mid` or `\Vert` inside a table cell | a bare `\|` in a table's maths (splits the cell) |
| `<span>$</span>100` for a literal dollar | `$100` next to maths on the same line |
| Unicode in prose and code comments (σ², Σ⁻¹, ⊗) | maths images |

GitHub has no custom macros, so expand them before filing:

```bash
python <skill folder>/scripts/expand_macros.py body.md
```

## MyST

1. Write the macro block into the project's `myst.yml`, under `project:`:

```bash
python <skill folder>/scripts/emit_macros.py myst
```

2. Label an equation with the directive form, or after the closing `$$` on the same line:

````markdown
```{math}
:label: eq-3dvar
\mathcal{J}(\vect{x}; \gvect{\theta}) = \frac{1}{2} \Vert \vect{y} - h(\vect{x}) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}
```
````

KaTeX cannot bold Latin and Greek letters with one macro (`\boldsymbol{\mathrm{#1}}` and `\boldsymbol{\mathbf{#1}}` each lose the bold on at least one alphabet), which is why each kind has a Latin and a Greek macro.

## mkdocs

1. Enable `pymdownx.arithmatex` with `generic: true` in `mkdocs.yml`.
2. Write the engine config into `docs/javascripts/`:

```bash
python <skill folder>/scripts/emit_macros.py mkdocs-mathjax --output docs/javascripts/mathjax.js
```

3. List that file before the engine in `extra_javascript` (for KaTeX, use `mkdocs-katex` and load KaTeX's `auto-render`):

```yaml
extra_javascript:
  - javascripts/mathjax.js
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
```

## Unicode in code comments and docstrings

Code comments carry the equation above the line that computes it (`style-code-sketch`), in plain Unicode; the shape column carries what bold carries in maths.

| LaTeX | Unicode | LaTeX | Unicode |
| --- | --- | --- | --- |
| `\mathbf{x}`, `\boldsymbol{\Sigma}` | x, Σ (no bold) | `^\top`, `^{-1}`, `^2` | ᵀ, ⁻¹, ² |
| `\mathbb{R}`, `\mathbb{E}`, `\mathbb{P}`, `\mathbb{V}` | ℝ, 𝔼, ℙ, 𝕍 | `\nabla`, `\partial`, `\int` | ∇, ∂, ∫ |
| `\sum`, `\prod` | ∑, ∏ (U+2211, U+220F, never Σ) | `\otimes`, `\odot`, `\circ` | ⊗, ⊙, ∘ |
| `\mathcal{L}`, `\mathcal{J}`, `\mathcal{N}`, `\mathcal{D}`, `\mathcal{T}` | ℒ, 𝒥, 𝒩, 𝒟, 𝒯 | `\in`, `\subset`, `\approx`, `\le`, `\sim` | ∈, ⊂, ≈, ≤, ~ |
| `\mid`, `\Vert` | ∣, ‖ | `\star`, `\dagger`, `\to`, `\mapsto` | ⋆, †, →, ↦ |
| `\hat{x}`, `\bar{x}`, `\tilde{x}`, `\dot{x}` | x̂, x̄, x̃, ẋ | `\times`, `\infty` | ×, ∞ |

- Subscripts: Unicode only for digits and the letters that exist (ₙ ₖ ₜ ₓ ₑ ᵢ ⱼ); otherwise ASCII, `Σ_ε`, `x_obs`, `μ_x`. Never ᵧ, which is a Greek gamma, not a y.
- No mathematical bold characters (𝐱, 𝚺): monospace fonts and terminals draw them unreliably.
- Code identifiers stay ASCII and descriptive (`style-python`); the Unicode lives in comments and docstrings only.
- ruff's RUF001–RUF003 flag ∣, ℝ, ℓ and ℍ (entropy) as confusable characters. Add them to `allowed-confusables` under `[tool.ruff.lint]` (gaussx does) rather than falling back to ASCII `|`.

```python
# Λ_x|y = Hᵀ Λ_ε H + Λ_x
posterior_precision = gauss_newton_term + prior_precision      # (D_x, D_x)
```

## Verified

29 Oct 2026, from scratch builds outside any repository:

- All 968 maths spans in this skill render without error in MathJax 3 (GitHub's package set, undefined macros made errors) and in KaTeX 0.16, and 14 macro test expressions render in both with the macros loaded.
- A MyST site (`myst` 1.9.0, book theme) with the `math:` block from `emit_macros.py` rendered all 12 test expressions with KaTeX and no `katex-error`, including tables with `\mid`, `\Vert` and `\vert`.
- An mkdocs site (`mkdocs` 1.6.1, `pymdownx` 10.21) with `arithmatex` kept all 12 expressions as maths and every table row at two cells.
- GitHub itself was not posted to; its rules come from `gh-issue` and the MathJax check above.

Approved for ensemble methods (gaussx `_inference`): ℰ the ensemble and |ℰ| its size, 𝒥 the cost, ζ multiplicative inflation, Λ_w, Σ_w and w̄ for ETKF weight space, H̃ observation anomalies, φ and ϑ latitude and longitude. All are in ruff `allowed-confusables` where gaussx uses them.
