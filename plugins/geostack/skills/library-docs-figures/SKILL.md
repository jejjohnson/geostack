---
name: library-docs-figures
description: The GeoStack standard for a library's documentation figures, set by argminax. Which figures a stack library's docs carry beyond the README's three (a code map of its modules and imports, a hub of what builds and reads its central object, a comparison of the ways to do one job, a gallery of the structures or families a module offers, a detail figure per module), where each goes in the docs, how each is drawn once in geostack (HTML rendered to PNG, or light and dark SVGs written from data), how every fact on a figure is traced to the code or the docs, and how the library's pages embed them from geostack's main. Use when drawing, placing, reviewing or updating the figures of a stack library's docs, or when a docs page explains in prose an idea that a figure would say faster.
---

# Library docs figures

Draft 1 · 18 Nov 2026 · taken from argminax's docs figures ([geostack#198](https://github.com/jejjohnson/geostack/pull/198), [geostack#200](https://github.com/jejjohnson/geostack/pull/200)) and the argminax pages that place them, which Eman named as the standard.

A library's README carries three figures: the overview, the role ladder and the dependencies (the `library-readme` skill in geostack's `.claude/skills/`, §10). Its docs carry those three and more: a figure wherever a page explains an idea that a picture says faster. This skill is for the docs. Every figure is drawn once in geostack, under `docs/images/libraries/<pkg>_<name>/`, and the library's pages embed it from geostack's `main`, so the stack's figures stay in one place and in one style.

It builds on three other skills: `role-ladder` (in this plugin) for the ladder figure, `eman-style-math` for the notation on every figure, and `library-readme` for the README's three.

## 1. The catalogue

Eight kinds. The first four belong in every library's docs; the other four appear only where the docs have the idea. One figure per idea, never one per page.

| Kind | Shows | Form | Goes on | argminax's |
|---|---|---|---|---|
| Overview | what you supply, the one idea, what the library does with it, what comes out | HTML → PNG | the landing page (and the README) | `argminax_overview/` |
| Role ladder | the public names on the rungs of the role ladder | light and dark SVGs | the architecture page (and the README) | `argminax_roles/` |
| Dependencies | who imports the library, which modules each uses, what it builds on | light and dark SVGs | the architecture page's dependencies (and the README) | `argminax_ecosystem/` |
| Code map | each public module, what it imports (dashed through an extra), and the imports between modules; and the exception hierarchy, with where each error is raised | light and dark SVGs | the architecture page, where the layers are described; the exceptions API page | `argminax_imports/`, `argminax_exceptions/` |
| Hub | the library's central object in the middle, what builds it on one side and what reads it on the other, each spoke labelled with the operation | HTML → PNG | the quickstart, after the core-idea table | `argminax_hessian/` |
| Comparison | the ways to do one job side by side, the same picture in each column, with what each costs and when it is right | HTML → PNG | the section that discusses the choice, and the API page of the module that owns it | `argminax_adjoints/` |
| Gallery | the structures or families a module offers, drawn as their shape, with storage and cost, and the special cases below | HTML → PNG | the maths or guide section that lists them, above its table | `argminax_blr_families/` |
| Module detail | one module's loop: what goes in, what it is made of, its state, what comes out | HTML → PNG | that module's guide | `argminax_blr/` |

The forms split by job. A figure that maps the code (ladder, dependencies, code map) is an SVG written from a data block, in light and dark, because the data changes as the code does. A figure that explains an idea (overview, hub, comparison, gallery, detail) is HTML rendered to a PNG, because it needs layout, equations and connectors. `references/figure-kinds.md` has the layout of each kind and which argminax file to copy.

## 2. Choosing a library's figures

1. **Read the docs in order.** List each section that explains a structure, a choice between methods, a flow or a set of variants, in prose or in a table.
2. **Match each to a kind.** Skip an idea the text already says in a short paragraph, and an idea that only one sentence leads into.
3. **Check the README's three are in the docs**: the overview on the landing page, the ladder and the dependencies on the architecture page.
4. **Replace every diagram drawn in the library's repo** (a mermaid block, an ASCII sketch, an image in `docs/assets/` other than the logo) by a geostack figure. Check the old one against the code first: argminax's mermaid import graph missed three edges.
5. **Propose the list with a placement for each before drawing**, then draw one figure at a time, show it, and wait for a go-ahead before the next.

## 3. What goes on a figure

- **Every fact is traced.** Formulas come from the docstrings or the page the figure sits on, names from `__all__` on `main`, edges from the `import` lines on `main`, costs from the docs' own tables. When a figure and the code disagree, the figure follows the code and the PR says so.
- **The figure summarises its section; it does not replace it.** Derivations and caveats stay in the text. A figure states a result in one line and names where it is derived.
- **Say exactly what is true.** "Its Gauss–Newton form", not "the Hessian", when the formula is the approximation; no "never" while one path does it (argminax's exact sampling factors the precision densely); "O(D R²) with a built-in curvature" when a callable costs more.
- **The library's notation.** The letters of the docstrings and the `eman-style-math` vocabulary, written in Unicode, since a PNG or an SVG cannot run MathJax. Use `<sup>` and `<sub>` for anything longer than one character; Unicode superscripts render too small.
- **A title and a subtitle.** The title is "<pkg> — <the idea in a few words>" ("argminax — one Hessian, every answer"); the subtitle is one sentence saying what to read off the figure.
- **Real names in code font**, each one public; module names on cards say which module owns the card.

## 4. Drawing them

Each figure is a folder under geostack's `docs/images/libraries/`, named `<pkg>_<name>/`, holding its source, its script and its output. Start from argminax's folder for the same kind and change the content, not the style.

- **HTML → PNG.** The source is `<pkg>_<name>.html` with `_base.css` and `_connect.js` copied unchanged; `render.py` (cyclopts and loguru) screenshots the `#diagram` element at 2x. Lanes take their colour from the classes in `_base.css` (`find`, `compute`, `cut`, `out`); `wire(from, to, {from, to, label, color, fromShift})` draws a labelled connector between two elements by id; a script before `ready()` may generate repeated shapes such as a matrix pictogram.
- **Light and dark SVGs.** `make_svgs.py` (cyclopts and loguru) holds the data at the top, one palette per theme, and writes `<pkg>_<name>-light.svg` and `-dark.svg`. Chips are sized from their text; rows wrap their chips at the right edge.
- **Render in Chromium.** Playwright's Chromium, with `CHROMIUM_PATH` for another build; where it cannot find `libgbm.so.1`, take it from conda-forge with `pixi exec -s libgbm` (each `render.py` docstring has the line). Preview the SVGs in Chromium in both themes: cairosvg mis-spaces bold `tspan` runs, so its previews are not a check.
- **Read every render at full size** for clipped labels, wrapped equations, overlapping connectors and text that runs past its box, and fix them before showing it.

Before the geostack PR: regenerate each output from the committed source and compare it byte for byte with the draft that was approved, and run ruff check and format on the scripts with geostack's config.

## 5. Placing them

Merge the geostack PR first. A library's docs build downloads remote images, and an image that is not yet on `main` fails a strict build. Then embed each figure from `main`:

````markdown
```{figure} https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/<pkg>_<name>/<pkg>_<name>.png
:alt: <what the figure shows, in one sentence>
:width: 100%
:label: fig-<name>

<One sentence on what it shows.> The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/<pkg>_<name>) with the stack's other library schematics.
```
````

- **An SVG pair** goes in as its light file: a MyST figure has one source. The README uses `<picture>` for both (`library-readme`, §10).
- **Put it next to the text it summarises**: after the paragraph that introduces the idea, or above the table it condenses. Never between a sentence and the next one that continues its thought.
- **An API page** built by mkdocs or zensical takes a Markdown image of the same URL, under the module's summary.
- **Build the docs.** Before the geostack PR merges, point the URLs at its branch, build, and point them back at `main`. Use the latest mystmd, as CI does (`npx -y mystmd`): an older local `myst` stops at the theme step. Check every image URL returns 200.

## 6. Keeping them current

A figure shows code, so it changes with the code. When a public name, an import, an equation or a cost changes, the figure that shows it changes in geostack in the same wave, and the library's PR links that geostack PR. The captions name the geostack folder, so a reviewer of a docs page can find the source of its figure.

## Checklist

- [ ] The overview, role ladder, dependencies and code map are in the docs; the other kinds where the docs have the idea, one figure per idea.
- [ ] No diagram is drawn in the library's repo; each old one was checked against the code before it was replaced.
- [ ] Every name is public, every edge is an import on `main`, every formula and cost is the docstring's or the page's.
- [ ] Each figure was shown and approved before the next was drawn.
- [ ] Every output regenerates byte for byte from its source; the scripts pass ruff; the SVGs were checked in Chromium in both themes.
- [ ] The geostack PR merged before the library's; every figure URL returns 200; the docs build strict with the latest mystmd.
- [ ] Each figure sits next to the text it summarises, with alt text and a caption that names its geostack folder.
