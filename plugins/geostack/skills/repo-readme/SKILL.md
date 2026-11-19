---
name: repo-readme
description: Eman's README standard for his JAX library repos (argminax, vardax and the rest of the GeoStack): centred header, one central equation, an overview schematic, highlights, the core-idea table, install, one runnable style-code-sketch quickstart, module and ecosystem tables with a role-ladder figure and a dependency figure drawn in geostack, with everything else moved to the docs quickstart. Use when writing, rebuilding or reviewing a library README, or drawing a library's schematics.
---

# Library README standard

Draft 2 · 18 Nov 2026 · argminax's README is the standard: its text from argminax PRs #76 and #82, and its three schematics (overview, role ladder, dependencies), which live in geostack. Draft 1 took the text from vardax's README and argminax PR #76.

A library README answers four questions in this order: what problem the library solves, what the one idea is, how to install it, and what a first run looks like. Everything else (derivations, second examples, variants, calling conventions, training loops) goes to the docs quickstart, and the README links there.

It builds on four other skills, which still apply in full: `eman-prose-tech` (docs register) for every sentence, `style-math` (GitHub target) for every symbol, `style-code-sketch` for every code block and `style-python` for the code inside it.

This standard is for libraries. A research project made from the GeoStack template keeps the template's README layout.

This is the only copy. It lives in geostack's plugin (`plugins/geostack/`), which installs it in every repository that adds the geostack marketplace, so change it here and nowhere else.

## Exemplars

argminax is the standard; read it first, then vardax. They show the two shapes the core-idea section takes: a set of questions about one object (argminax) and a family of methods (vardax).

- argminax: <https://github.com/jejjohnson/argminax/blob/main/README.md>, five questions about one minimiser, with all three schematics of §10.
- vardax: <https://github.com/jejjohnson/vardax/blob/main/README.md>, seven methods as specialisations of one cost. Its overview figure predates §10 and lives in its own repo; a rebuild moves it to geostack.

## 1. Section order

Use these sections, in this order. Sections marked optional appear only when the repo has the material; never write one to fill a slot.

| # | Section | Holds |
|---|---|---|
| 1 | Header (no heading) | logo, tagline, badges, link row, then a `---` rule |
| 2 | Opening (no heading) | the problem, the central equation, a short sketch, the "composed, not coded" paragraph |
| 3 | Overview figure | the overview schematic of §10 (a flow, or the operations figure for a toolbox library), embedded from geostack, with alt text that says what it shows |
| 4 | `## 🌟 Highlights` | five to seven bullets |
| 5 | `## 📐 <the core idea>` | one table and at most three display equations |
| 6 | `## 📥 Install` | the install command, the extras, the Python version |
| 7 | `## <emoji> Quickstart: <the example>` | one runnable example and what it printed |
| 8 | A second sketch (optional) | only when it shows a seam the quickstart does not (vardax's "Swap the method, keep the pipeline") |
| 9 | `## 🔬 What else is in the box` | the role-ladder figure of §10, then the module table |
| 10 | `## 📓 Tutorials` (optional) | one row per notebook or group of notebooks |
| 10a | `## 🖼️ Gallery` (optional) | two to four specimen plots of what the library computes, each under one sentence naming the call that drew it |
| 11 | `## 📖 Documentation` | one bullet per docs entry point |
| 12 | `## 🧩 Ecosystem` | the dependency figure of §10, then the owner table and the dependency direction |
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
3. **A sketch** of three to five lines in `style-code-sketch` form showing the library's main calls. It may use names the quickstart defines; its shapes header says so.
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
- **Prose before the block** defines every symbol the comments use, in `style-math` GitHub form.
- **The block follows `style-code-sketch`**: a shapes header, the equation above each mathematical line, a status marker on every call into the stack, the shape comment above bare assignments, einx for contractions, one operation per line, 88 columns.
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

## 10. The three schematics

Every library README carries three figures, drawn once in geostack and embedded from there, so the stack's schematics stay in one place and in one style. argminax's are the reference: copy its folders and change the data, not the layout.

| Figure | Shows | geostack folder | Made by | README section |
|---|---|---|---|---|
| Overview | what you supply, the one idea, what the library does with it, what comes out, and the libraries underneath; or, for a toolbox library, the one object and the operations on it | `docs/images/libraries/<pkg>_overview/` | `<pkg>_overview.html` (with `_base.css` and `_connect.js`), rendered to a PNG by `render.py` | under the opening (§1, row 3) |
| Role ladder | the public names on the rungs of the GeoStack role ladder | `docs/images/libraries/<pkg>_roles/` | `make_svgs.py`, light and dark SVGs | What else is in the box |
| Dependencies | who imports the library, which of its modules each uses, and what it builds on | `docs/images/libraries/<pkg>_ecosystem/` | `make_svgs.py`, light and dark SVGs | Ecosystem |

A detail figure for one module (argminax's `argminax_blr/`) follows the overview's form and sits in the docs guide for that module, not in the README. The docs carry these three and more; the `library-docs-figures` skill (`plugins/geostack/skills/library-docs-figures/`) is the standard for them.

**The overview** has four columns joined by labelled arrows, and a row of slabs underneath:

1. what the caller supplies (the cost, the inputs, the solver or the model);
2. the one idea the library is built around, with its equation (argminax: one minimum and the Hessian there; vardax: one cost);
3. what the library does with it, one card per question or method, each naming its module and public names;
4. what comes out, with the result the reader will use most drawn solid;

and one slab per upstream library the package extends, saying what it takes from it. The title says the idea in a few words ("argminax: five questions, one minimum").

**A toolbox library gets the operations figure instead.** When the library is one object with several operations, each taking different inputs and giving different outputs, supply and out lanes would invent a pipeline it does not have. Draw the object at the top with its equation, one card per operation below it (the verb, its equation, its module, its public names), and the library's lower modules as slabs underneath; it still lives in `<pkg>_overview/`. functax is the example (one expansion; interpolate, differentiate, integrate, project, transform). The `library-docs-figures` skill has its layout under "Operations".

**The gallery** holds specimens, not schematics: plots of what the library computes on toy data, drawn by the library's own `docs/assets/readme/figures.py` and embedded by relative path. Use it when a picture of the output proves something the text only claims (functax: Slepian functions leaking out of their cap past the Shannon number). The `library-docs-figures` skill has the rules under "Specimen".

**The role ladder** takes its rungs from the `role-ladder` skill (`plugins/geostack/skills/role-ladder/` here):

- One row per rung the library has, highest first, each labelled with its number, its role and its grammar word ("4 algorithm · the sentence") and a caption saying what that rung is in this library. Never label rows L0, L1: the role is the point.
- The chips are public names from `__all__` on `main`, placed as on the library's skill-shelf page (`docs/design/skill_tree_<pkg>.md`). A name the shelf page does not place yet is placed by the skill's §2 test, and the PR says which names were placed that way.
- A rung the library leaves to another library is drawn dashed with that library's names ("none of argminax's own: one step is gaussx's"). Rungs above the library's top are left out.
- A third-party floor the library extends goes in a dashed side box with an arrow to the rung it plugs into (argminax: optimistix, into the strategies).
- A "built on" row closes the figure.

**The dependency figure** has the library at the top with its icon, then the stack layers as boxes:

- One card per package that imports the library, with what it is and the library's modules it uses, read from its `from <pkg>… import` lines on `main`.
- A package that imports it only through an extra is drawn dashed, with the extra named under it; an edge between two dependents is a grey arrow.
- A strip underneath lists what the library itself builds on.
- The edges come from geostack's `docs/api/dependency_graph.md`. When a package's `main` and the graph disagree, draw `main` and say so in the PR.

**Maths on them.** The overview's equations are typeset with KaTeX, as on every HTML figure, so they read as the README's maths and not as code; the SVGs write maths in Unicode. The `library-docs-figures` skill (§3) has the rule and `functax_overview/` is the example.

**Making them.** The scripts use cyclopts and loguru and run with `uv run --no-project --with …` as their docstrings say. Render every figure in Chromium before committing, both themes for the SVGs: cairosvg mis-spaces bold `tspan` runs, so its previews are not a check. The overview's PNG needs Chromium anyway; where it cannot find `libgbm.so.1`, take it from conda-forge with `pixi exec -s libgbm`.

**Embedding them.** Merge the geostack PR first: the library's docs build downloads remote images, and a 404 fails a strict build. Then embed from geostack's `main`:

```html
<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/<pkg>_overview/<pkg>_overview.png" alt="<what it shows>" width="100%"></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/<pkg>_roles/<pkg>_roles-dark.svg">
    <img alt="<what it shows>" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/<pkg>_roles/<pkg>_roles-light.svg" width="100%">
  </picture>
</p>
```

The dependency figure is embedded like the role ladder. The library's design page in geostack shows all three with relative paths.

## 11. Maths and prose

- Maths is the `style-math` GitHub target: no macros, inline `` $`…`$ ``, display in a `math` fence, `\mid` inside table cells, one expression per line. The letters match the package's docstrings.
- Prose is the `eman-prose-tech` docs register: "we" for the maintainers, plain words, parentheses for glosses, no em dashes. One paragraph per line, never hard-wrapped.
- No ASCII box-and-arrow diagrams; a figure is one of the schematics of §10, an image from `docs/assets/` (the logo) or a Mermaid block.

## 12. Verification

Run these before the README ships:

1. Extract every Python block and run the runnable ones against the repo's `main`, in float64 if the package's tests use it. Paste the printed output into the "This prints" sentence.
2. Lint and format each block with the repo's own ruff config, by passing it on stdin with a path inside the repo: `ruff check --stdin-filename docs/readme_block.py < block.py` (ruff's ambiguous-Unicode rule, RUF003, may be ignored for maths comments).
3. Check the maths with `check_math.py` from the `style-math` skill, on the README with `--target github` and on any docs page the change touched with `--target myst`. The script declares its dependencies inline, so `uv run` installs them: `uv run ~/.claude/skills/style-math/scripts/check_math.py README.md --target github`.
4. Check the prose with `style_check.py` from the `eman-prose-tech` skill, which needs only the standard library: `python ~/.claude/skills/eman-prose-tech/scripts/style_check.py prose.md --register docs`, on the prose with code blocks and HTML removed. Tables and the reference list skew its sentence metrics, so read the flags instead of chasing the bands.
5. Check every name in the tables and on the figures against `__all__`, every URL (the figure URLs included) with `curl -s -o /dev/null -w '%{http_code}'`, and every status claim ("ships", "Wave 4") against the live issue.
6. Build the docs when the change touched a docs page.

Both scripts come with the `style-math` and `eman-prose-tech` skills. Without them, do steps 3 and 4 by hand against §11 (no macros, inline `` $`…`$ ``, display maths in `math` fences, `\mid` in table cells, no em dashes, no hard wraps), and say in the PR that the two scripts were not run.

## Checklist

- [ ] Sections in the order of §1; optional ones only where the material exists.
- [ ] The tagline, the opening and the ecosystem table name every upstream library the package extends.
- [ ] One central equation, every symbol defined with its shape.
- [ ] Highlights traceable to public names; no unmeasured claims.
- [ ] One runnable quickstart in `style-code-sketch` form, with its real printed output.
- [ ] Derivations, later examples and training loops moved to the docs quickstart, which uses the same notation.
- [ ] Every table name in `__all__`, every URL 200, every status claim checked live.
- [ ] The three schematics of §10 drawn in geostack, rendered in Chromium in both themes, and embedded from geostack's `main` after the geostack PR merged; every chip in `__all__`, every edge in the dependency graph or on `main`.
- [ ] `check_math.py --target github` clean (or the maths checked by hand against §11 when the script is not installed, and the PR says so); prose in the docs register, no em dashes, no hard wraps.

## Open questions

- Should the README carry the quickstart at all once the docs site is live, or only the opening sketch? Both exemplars keep it.
- Should optional sections (tutorials, a second sketch) have a size threshold, such as three or more notebooks?
- Should the role-ladder and dependency scripts become one shared generator that reads each library's data, instead of a copy per library? Two libraries' worth of copies would show what varies.
