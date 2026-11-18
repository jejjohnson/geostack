---
name: library-readme
description: Eman's README standard for his JAX library repos (vardax, argminax and the rest of the GeoStack): centred header, one central equation, highlights, the core-idea table, install, one runnable code-sketch quickstart, module and ecosystem tables, with everything else moved to the docs quickstart. Use when writing, rebuilding or reviewing a library README.
---

# Library README standard

Draft 1 · 18 Nov 2026 · taken from the vardax README and argminax PR #76, which Eman named as the standard.

A library README answers four questions in this order: what problem the library solves, what the one idea is, how to install it, and what a first run looks like. Everything else (derivations, second examples, variants, calling conventions, training loops) goes to the docs quickstart, and the README links there.

It builds on four other skills, which still apply in full: `eman-prose-tech` (docs register) for every sentence, `eman-style-math` (GitHub target) for every symbol, `code-sketch` for every code block and `python-style` for the code inside it.

This standard is for libraries. A research project made from the GeoStack template keeps the template's README layout.

The canonical copy lives in [jejagents](https://github.com/jejjohnson/jejagents) (`skills/library-readme/`); this copy is here so an agent working on a stack library from GeoStack finds it. Change both together.

## Exemplars

Read both before writing. They show the two shapes the core-idea section takes: a family of methods (vardax) and a set of questions about one object (argminax).

- vardax: <https://github.com/jejjohnson/vardax/blob/main/README.md>, seven methods as specialisations of one cost.
- argminax: <https://github.com/jejjohnson/argminax/blob/main/README.md> (rebuilt in PR #76), three questions about one minimiser.

## 1. Section order

Use these sections, in this order. Sections marked optional appear only when the repo has the material; never write one to fill a slot.

| # | Section | Holds |
|---|---|---|
| 1 | Header (no heading) | logo, tagline, badges, link row, then a `---` rule |
| 2 | Opening (no heading) | the problem, the central equation, a short sketch, the "composed, not coded" paragraph |
| 3 | Overview figure (optional) | one diagram from `docs/assets/`, with alt text that says what it shows |
| 4 | `## 🌟 Highlights` | five to seven bullets |
| 5 | `## 📐 <the core idea>` | one table and at most three display equations |
| 6 | `## 📥 Install` | the install command, the extras, the Python version |
| 7 | `## <emoji> Quickstart: <the example>` | one runnable example and what it printed |
| 8 | A second sketch (optional) | only when it shows a seam the quickstart does not (vardax's "Swap the method, keep the pipeline") |
| 9 | `## 🔬 What else is in the box` | the module table |
| 10 | `## 📓 Tutorials` (optional) | one row per notebook or group of notebooks |
| 11 | `## 📖 Documentation` | one bullet per docs entry point |
| 12 | `## 🧩 Ecosystem` | the owner table and the dependency direction |
| 13 | `## 🌳 Repository layout` | a short plain-text tree |
| 14 | `## ⚙️ Development` | the repo's own task commands |
| 15 | `## 🎓 References` | the papers the docs and docstrings already cite |
| 16 | `## 📄 License` | one line |

Each heading starts with one emoji. Headings are plain names, never "Title: Subtitle", except the quickstart, which names its example ("Quickstart: 4DVar on Lorenz-63").

## 2. The header

```html
<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="<pkg>" src="docs/assets/logo-light.svg" width="340">
</picture>

**<One sentence: what it does, for JAX, and what it is built on.>**

<badges, one per line>

📖 [Documentation](<site>) ·
🚀 [Quickstart](<site>/quickstart/) ·
📚 [API reference](<site>/reference/) ·
🧭 [Design](<design page>)

</div>

---
```

- **The tagline names every upstream library the package extends.** A library that extends optimistix and optax says both ("on optimistix, optax and gaussx"). argminax's first README named optimistix and gaussx but left optax to its training section, and that is the gap this rule closes.
- **Badges**, in this order and only for workflows that exist in `.github/workflows/`: Tests, Lint, Type Check, Docs, codecov, Python version, License, Ruff, uv, pre-commit.
- **The link row** has one entry per docs entry point that exists (a tutorials index, a maths reference, a guide). Check every URL returns 200.

## 3. The opening

Two to four short paragraphs with no heading, in this order:

1. **The problem**, in the reader's terms, before the library's name: "A retrieval is not finished when the optimiser stops." Then what the library treats as its central object.
2. **The central equation** in a `math` fence, with every symbol defined and given its shape in the sentence after it.
3. **A sketch** of three to five lines in `code-sketch` form showing the library's main calls. It may use names the quickstart defines; its shapes header says so.
4. **Composed, not coded**: one paragraph naming each upstream library with a link and saying what the package takes from it ("its solvers are optimistix solvers, so `optx.minimise` keeps working").

## 4. Highlights

Five to seven bullets, each `- <emoji> **<Label>**: <one sentence>.` Each bullet states a capability that ships today and can be traced to a public name. No adjectives without a fact behind them, and no claim about speed or scale unless a benchmark in the repo measured it.

## 5. The core idea

The section that makes the library one idea instead of a list of functions. Name the heading after the idea ("The seven methods", "Three questions about one minimiser").

- **One table** that maps the idea onto the modules: methods, controls and when to reach for each (vardax), or questions, equations and modules (argminax).
- **At most three display equations**, each followed by "where ..." or "with ..." prose that defines what is new.
- **The derivation stays in the docs.** State the result and cite where it comes from; the quickstart derives it.

## 6. Install

- Say whether it is on PyPI, what it brings in, and the Python version (from `requires-python`).
- One `pip install` block for the base package.
- One sentence listing every extra in `[project.optional-dependencies]` and what each adds, then one block for the most-used extra.
- When the package has git dependencies, give the consumer's `[tool.uv.sources]` block, as vardax does, and say why uv needs it.

## 7. The quickstart

- **One example**, small enough to run on a laptop in seconds, with toy data built in the block.
- **Prose before the block** defines every symbol the comments use, in `eman-style-math` GitHub form.
- **The block follows `code-sketch`**: a shapes header, the equation above each mathematical line, a status marker on every call into the stack, the shape column on bare assignments, einx for contractions, one operation per line, 88 columns.
- **It runs as written** and ends with one `print`.
- **"This prints `...`"** follows the block, with the real output pasted from a run, and a sentence that reads each number.
- **One closing paragraph** says what transfers ("swap `forward` for a radiative transfer code ...") and links the docs quickstart for the rest.

## 8. What goes to the docs quickstart

Move these out of the README and into `docs/guide/quickstart.md` (or the guide that owns them), and link them from the README:

- derivations and "where it stops holding" caveats;
- the calling convention and the precision advice;
- every example after the first, including variants of the first (an outer loss, a `vmap` over a swath, a different adjoint);
- training-loop and optimiser examples;
- tables of numbers from those examples.

The docs page keeps the same notation as the README. When the move touches a page in older notation, convert its maths to the house notation in the same change and say so in the PR.

## 9. The tables at the bottom

- **What else is in the box**: `Module | Symbols | Notes`, or `Area | Symbols | Notes` when the package has one public module. Every name comes from the module's `__all__` on `main`.
- **Tutorials**: `# | Tutorial | What it shows`, linking each notebook.
- **Documentation**: bold link, colon, what the reader finds there.
- **Ecosystem**: `Concern | Owner`, one row per upstream library the package extends or composes, with links. Below it, one sentence on the dependency direction and the consumers, with their live status ("their switch-over is Wave 4 (#12)"), checked against the issue before writing.
- **Repository layout**: plain text, the public modules and the top-level folders only, with `←` comments.
- **Development**: the repo's own task runner (`make`, `uv run poe`), copied from `AGENTS.md`, then a pointer to `AGENTS.md`.
- **References**: author, year, linked title, venue in italics. Use only papers cited in the repo's docs or docstrings, and take venues and identifiers from there.

## 10. Maths and prose

- Maths is the `eman-style-math` GitHub target: no macros, inline `` $`…`$ ``, display in a `math` fence, `\mid` inside table cells, one expression per line. The letters match the package's docstrings.
- Prose is the `eman-prose-tech` docs register: "we" for the maintainers, plain words, parentheses for glosses, no em dashes. One paragraph per line, never hard-wrapped.
- No ASCII box-and-arrow diagrams; a figure is an image from `docs/assets/` or a Mermaid block.

## 11. Verification

Run these before the README ships:

1. Extract every Python block and run the runnable ones against the repo's `main`, in float64 if the package's tests use it. Paste the printed output into the "This prints" sentence.
2. Lint and format each block with the repo's own ruff config, by passing it on stdin with a path inside the repo: `ruff check --stdin-filename docs/readme_block.py < block.py` (ruff's ambiguous-Unicode rule, RUF003, may be ignored for maths comments).
3. Check the maths with `check_math.py` from the `eman-style-math` skill, on the README with `--target github` and on any docs page the change touched with `--target myst`. The script declares its dependencies inline, so `uv run` installs them: `uv run ~/.claude/skills/eman-style-math/scripts/check_math.py README.md --target github`.
4. Check the prose with `style_check.py` from the `eman-prose-tech` skill, which needs only the standard library: `python ~/.claude/skills/eman-prose-tech/scripts/style_check.py prose.md --register docs`, on the prose with code blocks and HTML removed. Tables and the reference list skew its sentence metrics, so read the flags instead of chasing the bands.
5. Check every name in the tables against `__all__`, every URL with `curl -s -o /dev/null -w '%{http_code}'`, and every status claim ("ships", "Wave 4") against the live issue.
6. Build the docs when the change touched a docs page.

Both scripts come with the personal skills in [jejagents](https://github.com/jejjohnson/jejagents): clone it and run its `./install.sh`, which copies the skills into `~/.claude/skills/`. Without them, do steps 3 and 4 by hand against §10 (no macros, inline `` $`…`$ ``, display maths in `math` fences, `\mid` in table cells, no em dashes, no hard wraps), and say in the PR that the two scripts were not run.

## Checklist

- [ ] Sections in the order of §1; optional ones only where the material exists.
- [ ] The tagline, the opening and the ecosystem table name every upstream library the package extends.
- [ ] One central equation, every symbol defined with its shape.
- [ ] Highlights traceable to public names; no unmeasured claims.
- [ ] One runnable quickstart in `code-sketch` form, with its real printed output.
- [ ] Derivations, later examples and training loops moved to the docs quickstart, which uses the same notation.
- [ ] Every table name in `__all__`, every URL 200, every status claim checked live.
- [ ] `check_math.py --target github` clean (or the maths checked by hand against §10 when the script is not installed, and the PR says so); prose in the docs register, no em dashes, no hard wraps.

## Open questions

- Should the README carry the quickstart at all once the docs site is live, or only the opening sketch? Both exemplars keep it.
- Should optional sections (tutorials, a second sketch) have a size threshold, such as three or more notebooks?
