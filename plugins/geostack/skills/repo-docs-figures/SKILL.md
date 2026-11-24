---
name: repo-docs-figures
description: The GeoStack standard for a library's documentation figures, set by argminax, vardax and functax. Which figures a stack library's docs carry beyond the README's three (an operations figure in place of the overview for a toolbox library, a code map of its modules and imports, a hub of what builds and reads its central object, a comparison of the ways to do one job, a gallery of the structures or families a module offers, a detail figure per module, a pipeline for an end-to-end example, specimen plots of what the code computes), where each goes in the docs, how each is drawn (once in geostack as HTML rendered to PNG or light and dark SVGs written from data, or, for a specimen, by the library's own docs code), how every fact on a figure is traced to the code or the docs, and how the library's pages embed them from geostack's main. Use when drawing, placing, reviewing or updating the figures of a stack library's docs, or when a docs page explains in prose an idea that a figure would say faster.
---

# Library docs figures

Draft 4 · 19 Nov 2026 · a specimen typesets the maths in its labels, and a number the page quotes but never computes is fixed on the page. Taken from argminax's docs figures ([geostack#198](https://github.com/jejjohnson/geostack/pull/198), [geostack#200](https://github.com/jejjohnson/geostack/pull/200)) and the argminax pages that place them, which Eman named as the standard, and from vardax's ([geostack#204](https://github.com/jejjohnson/geostack/pull/204) to [geostack#211](https://github.com/jejjohnson/geostack/pull/211)), which add the pipeline and the forms a large library needs, and from functax's, which add the operations figure and the specimen.

A library's README carries three figures: the overview, the role ladder and the dependencies (the `repo-readme` skill in this plugin, §10). Its docs carry those three and more: a figure wherever a page explains an idea that a picture says faster. This skill is for the docs. Every schematic is drawn once in geostack, under `docs/images/libraries/<pkg>_<name>/`, and the library's pages embed it from geostack's `main`, so the stack's figures stay in one place and in one style. A specimen is the exception: it is a plot of what the library computes, so the library's own docs code draws it.

It builds on three other skills: `role-ladder` (in this plugin) for the ladder figure, `style-math` for the notation on every figure, and `repo-readme` (in this plugin) for the README's three.

## 1. The catalogue

Eleven kinds. The overview (or the operations figure in its place), the role ladder, the dependencies and the code map belong in every library's docs; the others appear only where the docs have the idea. One figure per idea, never one per page.

| Kind | Shows | Form | Goes on | Copy from |
|---|---|---|---|---|
| Overview | what you supply, the one idea, what the library does with it, what comes out | HTML → PNG | the landing page (and the README) | `argminax_overview/`, `vardax_overview/` |
| Operations | one object, one card per operation on it (the verb, its equation, the module that owns it and its public names), and the library's lower modules as a floor | HTML → PNG | in place of the overview, for a library with no single flow | `functax_overview/` |
| Role ladder | the public names on the rungs of the role ladder | light and dark SVGs | the architecture page (and the README) | `argminax_roles/`; `vardax_roles/` for a large library |
| Dependencies | who imports the library, which modules each uses, what it builds on | light and dark SVGs | the architecture page's dependencies (and the README) | `argminax_ecosystem/`; `vardax_ecosystem/` when every import is through an extra |
| Code map | each public module, what it imports (dashed through an extra), and the imports between modules; and the exception hierarchy, with where each error is raised | light and dark SVGs | the architecture page, where the layers are described; the exceptions API page | `argminax_imports/`, `argminax_exceptions/`; `vardax_imports/` for a large library |
| Hub | the library's central object in the middle, what builds it on one side and what reads it on the other, each spoke labelled with the operation | HTML → PNG | the quickstart, after the core-idea table | `argminax_hessian/` |
| Comparison | the ways to do one job side by side, the same picture in each column, with what each costs and when it is right | HTML → PNG | the section that discusses the choice, and the API page of the module that owns it | `argminax_adjoints/`, `vardax_adjoints/` |
| Gallery | the structures or families a module offers, drawn as their shape, with storage and cost, and the special cases below | HTML → PNG | the maths or guide section that lists them, above its table | `argminax_blr_families/`; `vardax_posterior/` for families sketched by what they do |
| Module detail | one module's loop: what goes in, what it is made of, its state, what comes out | HTML → PNG | that module's guide | `argminax_blr/` |
| Pipeline | an end-to-end example: its inputs, operators, method and hand-off, each box marked with its status; a band for the slot that varies, with what fills it on `main`; the cycle the example sits in | HTML → PNG | the example page, in place of its data-flow sketch | `vardax_methane/` |
| Specimen | what the library computes on toy data: a few basis functions, a field and its map, a member of each family, drawn as the real output | matplotlib → PNG, in the library's repo | the guide whose code it runs; a short gallery in the README | functax's `docs/assets/readme/figures.py` |

The forms split by job. A figure that maps the code (ladder, dependencies, code map) is an SVG written from a data block, in light and dark, because the data changes as the code does. A figure that explains an idea (overview, operations, hub, comparison, gallery, detail, pipeline) is HTML rendered to a PNG, because it needs layout, equations and connectors. A figure that shows what the code computes (specimen) is a plot drawn by running the library, because no drawing can prove what the library returns. `references/figure-kinds.md` has the layout of each kind and which folder to copy.

The logo and icon are not figures of this kind. They stay in the library's own `docs/assets/` (`logo-light.svg`, `logo-dark.svg`, `icon.svg`), drawn from data by a `make_assets.py` there, because the README header and `myst.yml` reference them by path; vardax's and argminax's are the examples. The dependency figure copies the icon into its geostack folder unchanged. Specimens stay in the library too, next to the script that draws them, because that script imports the library and runs its docs pages.

## 2. Choosing a library's figures

1. **Read the docs in order.** List each section that explains a structure, a choice between methods, a flow or a set of variants, in prose or in a table.
2. **Match each to a kind.** Skip an idea the text already says in a short paragraph, and an idea that only one sentence leads into. A section that shows what a function returns (a basis, a family member, the map an example produces) gets a specimen when the plot proves something the prose only states, such as Slepian functions leaking out of their cap past the Shannon number.
3. **Choose the overview's shape.** A library with one flow (you supply a cost, it finds a minimum, you read answers off it) gets the four-lane overview. A toolbox around one object, whose operations each take different inputs and give different outputs, gets the operations figure instead, because supply and out lanes would invent a pipeline the library does not have. functax (one expansion: interpolate, differentiate, integrate, project, transform) is the example; gaussx (one operator: solve, log-determinant, sample) and finitevolX (one grid) have the same shape.
4. **Check the README's three are in the docs**: the overview on the landing page, the ladder and the dependencies on the architecture page.
5. **Replace every diagram drawn in the library's repo** (a mermaid block, an ASCII sketch, an image in `docs/assets/` other than the logo and the specimens) by a geostack figure. Check the old one against the code first: argminax's mermaid import graph missed three edges, and vardax's methane data flow named a reader and a forward model that do not exist. A figure already in the stack's style moves to geostack instead of being redrawn, but its facts are checked like a new one's: vardax's overview carried the old notation, two deprecated names and two claims the code no longer supports.
6. **Propose the list with a placement for each before drawing**, then draw one figure at a time, show it, and wait for a go-ahead before the next.

## 3. What goes on a figure

- **Every fact is traced.** Formulas come from the docstrings or the page the figure sits on, names from `__all__` on `main`, edges from the `import` lines on `main`, costs from the docs' own tables. When a figure and the code disagree, the figure follows the code and the PR says so.
- **The figure summarises its section; it does not replace it.** Derivations and caveats stay in the text. A figure states a result in one line and names where it is derived.
- **Say exactly what is true.** "Its Gauss–Newton form", not "the Hessian", when the formula is the approximation; no "never" while one path does it (argminax's exact sampling factors the precision densely); "O(D R²) with a built-in curvature" when a callable costs more.
- **The library's notation, typeset as maths.** The letters of the docstrings and the `style-math` vocabulary (bold vectors and matrices, as the README writes them). An HTML figure typesets its maths with KaTeX, pinned (`katex@0.16.11` from jsdelivr), because Chromium runs it before the screenshot and a mono font makes equations look like code: wrap each formula in `<span class="m">`, typeset before drawing the wires, and call `ready()` after `document.fonts.ready`. Scope size rules under `#diagram` so they beat the global `.katex` rule, and wrap a long formula in braces so KaTeX does not break it. An SVG cannot run KaTeX, so it writes maths in Unicode, with `<sup>` and `<sub>` for anything longer than one character, since Unicode superscripts render too small. `functax_overview/` is the example.
- **A title and a subtitle.** The title is "<pkg> — <the idea in a few words>" ("argminax — one Hessian, every answer"); the subtitle is one sentence saying what to read off the figure.
- **Real names in code font**, each one public; module names on cards say which module owns the card. A deprecated alias is not a public name: vardax's `KStepAdjoint` is argminax's `KStep`.
- **A pictogram shows what the option does.** Draw each family or method acting on the same small object, not a table of numbers in cells: vardax's posterior gallery first shaded six cells by their variances and read as meaningless, and its second draft sketches each family in a two-dimensional state. Geometry on a sketch is computed from the algebra it shows (the update, the inverse curvature, the solver's steps), not drawn by eye, and the subtitle says the sketch is schematic.
- **A pipeline marks every box with its status**, in the `style-code-sketch` markers (today, today*, your code, not pinned), so the reader can tell what ships from what the page sketches. It follows the page's code, not its old diagram.
- **A specimen runs the docs' code as written.** Its script takes the page's Python blocks verbatim and in order, runs them in one namespace as a reader would, and draws from what they computed; it logs the numbers the page quotes, so a drift between the page and the code stops the figure. A number the page quotes but none of its blocks computes is a fault of the page: add the line to the page's block, so the script still only runs what a reader runs, and say so in the PR. Toy or subsampled data only, so it runs in about a minute on a laptop CPU. When the pages switch on float64 in a block, the script switches it on at the top too, so a figure drawn alone matches a full run: functax's Slepian modes with λ ≈ 1 tie in float32 and come out in another order. A specimen names in its subtitle the public call that drew it, with its arguments, written as code (`RBF(lengthscale=0.2)`, not "RBF, ℓ = 0.2"); a specimen of a basis calls the public constructor directly.
- **A specimen states its scales.** Each panel gets a colour bar, or the subtitle says the panels share one or are each scaled to their own maximum. A diverging map is symmetric about zero. Panel titles name the quantity in the library's notation and its unit, typeset with matplotlib's mathtext (`r"$\mu_*(x)$"`), never left as raw `μ_*` or `x^2`; a number in a title says what it is ("RMS on the observed pixels", not "1.27 / 1.26").
- **An import is lazy only if every import of it is.** Grep for module-level imports before drawing an edge as made inside a function: vardax's methods import the posterior inside functions, but `OptimalInterpolation` imports it at module level.

## 4. Drawing them

Each figure is a folder under geostack's `docs/images/libraries/`, named `<pkg>_<name>/`, holding its source, its script and its output. Start from the folder the catalogue names for the same kind and change the content, not the style.

- **HTML → PNG.** The source is `<pkg>_<name>.html` with `_base.css` and `_connect.js` copied unchanged; `render.py` (cyclopts and loguru) screenshots the `#diagram` element at 2x. Lanes take their colour from the classes in `_base.css` (`find`, `compute`, `cut`, `out`); `wire(from, to, {from, to, label, color, fromShift})` draws a labelled connector between two elements by id; a script before `ready()` may generate repeated shapes such as a matrix pictogram.
- **Light and dark SVGs.** `make_svgs.py` (cyclopts and loguru) holds the data at the top, one palette per theme, and writes `<pkg>_<name>-light.svg` and `-dark.svg`. Chips are sized from their text; rows wrap their chips at the right edge.
- **Matplotlib → PNG (specimens).** One `figures.py` (cyclopts and loguru) in the library's `docs/assets/`, run with the library's dev group plus matplotlib, writes every specimen at 150 dpi next to itself and takes `--only <name>` to redraw one. Give each figure a suptitle ("<pkg> — <the idea>") and a subtitle line, leave room above each row of panels for its titles (two rows of Mollweide maps need more height than `tight_layout` takes on its own), and use the same colour maps across the library's specimens: `RdBu_r` for signed fields, `viridis` for amounts.
- **Render in Chromium.** Playwright's Chromium, with `CHROMIUM_PATH` for another build; where it cannot find `libgbm.so.1`, take it from conda-forge with `pixi exec -s libgbm` (each `render.py` docstring has the line). Preview the SVGs in Chromium in both themes: cairosvg mis-spaces bold `tspan` runs, so its previews are not a check.
- **Read every render at full size** for clipped labels, wrapped equations, overlapping connectors and text that runs past its box, and fix them before showing it.
- **Line up the rows across columns.** In a comparison or a gallery give the picture, its caption, the prose line and the equation box fixed heights, so the same row starts at the same height in every column; keep lane headings to one line.
- **A large library groups its names.** The ladder groups each rung's chips under short labels with a dotted divider between groups, and puts what sits beside the ladder (a borrowed floor, setup functions) in dashed side boxes with an arrow to their rung (`vardax_roles/`). The code map groups the `_src` areas into cards by job, draws arrows only for imports between cards, and names the areas every card imports in a strip (`vardax_imports/`). When every name comes from the package root, the dependency cards show areas and the methods called instead of modules (`vardax_ecosystem/`). A ladder that would run past about 1800 px shows the main names instead of all of them: a family of evaluators becomes one chip for its submodule (`basis.sphere`), a long group of utilities keeps three to six representative names, and the PR lists every name left off (`functax_roles/`, 172 names in 1411 px). A legend shows only the marks the figure uses.
- **Three pitfalls.** `display: flex` on a block that holds `<sub>` or `<sup>` breaks them, so give it a `min-height` instead; `_base.css` styles `.step` and other generic names, so prefix a figure's own classes; and a Playwright page made with `set_content` cannot load a `file://` SVG, so preview SVGs through an HTML file on disk.

Before the geostack PR: regenerate each output from the committed source and compare it byte for byte with the draft that was approved, and run ruff check and format on the scripts with geostack's config. A specimen is checked in the library's PR instead: rerun `figures.py`, compare the logged numbers with the page, and run the library's own ruff.

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

- **A specimen** is embedded by a relative path from the library's `docs/assets/`, in the docs and the README alike, since it lives in the same repo and needs no geostack PR.
- **An SVG pair** goes in as its light file: a MyST figure has one source. The README uses `<picture>` for both (`repo-readme`, §10).
- **Put it next to the text it summarises**: after the paragraph that introduces the idea, or above the table it condenses. Never between a sentence and the next one that continues its thought.
- **An API page** built by mkdocs or zensical takes a Markdown image of the same URL, under the module's summary.
- **Build the docs.** Before the geostack PR merges, point the URLs at its branch, build, and point them back at `main`. Use the latest mystmd, as CI does (`npx -y mystmd`): an older local `myst` stops at the theme step. Check every image URL returns 200.

## 6. Keeping them current

A figure shows code, so it changes with the code. When a public name, an import, an equation or a cost changes, the figure that shows it changes in geostack in the same wave, and the library's PR links that geostack PR. The captions name the geostack folder, so a reviewer of a docs page can find the source of its figure. A specimen keeps itself current: the page's code changes, `figures.py` reruns it, and the library's PR carries the new PNG.

## Checklist

- [ ] The overview (or, for a toolbox library, the operations figure), role ladder, dependencies and code map are in the docs; the other kinds where the docs have the idea, one figure per idea.
- [ ] No diagram is drawn in the library's repo (specimens excepted); each old one was checked against the code before it was replaced.
- [ ] Every name is public, every edge is an import on `main`, every formula and cost is the docstring's or the page's.
- [ ] Every pictogram shows what its option does, with any geometry computed from the algebra; every pipeline box carries its status.
- [ ] Every specimen runs the page's code verbatim, logs the numbers the page quotes (each computed by a block of the page), typesets the maths in its labels, and states its colour scales.
- [ ] Each figure was shown and approved before the next was drawn.
- [ ] Every output regenerates byte for byte from its source; the scripts pass ruff; the SVGs were checked in Chromium in both themes.
- [ ] The geostack PR merged before the library's; every figure URL returns 200; the docs build strict with the latest mystmd.
- [ ] Each figure sits next to the text it summarises, with alt text and a caption that names its geostack folder.
