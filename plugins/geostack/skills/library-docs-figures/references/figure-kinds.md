# The figure kinds

One section per kind: what goes where on the canvas, what to read from the library, and the folder to copy from geostack's `docs/images/libraries/`. Every HTML figure is 1560 px wide and rendered at 2x; every SVG is 960 px wide; a specimen is about 9 to 13 inches wide at 150 dpi.

## Overview

Four lanes left to right, joined by labelled `wire` connectors, and a row of slabs underneath.

1. **supply**: what the caller writes (the cost and its inputs, the solver, the model or the loop), one card each.
2. **the one idea**: the object the library is built around, with its equation in a mono box and one or two labelled terms (argminax: the minimum, the retrieval cost, the Hessian there).
3. **what it does**: one card per question or method, each naming its module in accent and its public names in code font.
4. **out**: what comes back, with the result the reader uses most drawn solid (`node solid`).

The slabs name each upstream library the package extends and what it takes from it. Read the README's core-idea table and its highlights; the figure says the same in pictures. Copy `argminax_overview/`.

## Operations

The overview for a toolbox library, one with no single flow. The one object sits at the top as a wide box holding its equation and a line defining each symbol. Below it, one card per operation in a row, each joined to the object by a `wire` labelled with what the operation takes from it (values at points, coefficients, the field and its derivative). A card has the verb in accent, its equation in a mono line, the module that owns it, and its main public names in code font. Under the row, one full-width slab per lower module of the library that every operation uses (functax: `basis`, the basis functions φ; `geometry`, where x lives), each with a few of its names, and a last slab for the upstream libraries. Read the README's module table: each operation is a row there. Copy `functax_overview/`.

## Role ladder

One row per rung the library has, highest first, from the `role-ladder` skill: a numbered circle, the role in bold and its grammar word muted ("4 algorithm · the sentence"), a caption on the right saying what that rung is in this library, and the public names as chips that wrap. A rung the library leaves to another library is dashed and holds that library's names; a third-party floor sits in a dashed side box with an arrow to the rung it plugs into; a "built on" row closes it. Place each name as the library's skill-shelf page does (`docs/design/skill_tree_<pkg>.md` here), and by the skill's §2 test where the page has no row yet. Copy `argminax_roles/`. A library with a hundred names or more groups each rung's chips under short italic labels with a dotted divider between groups, and adds a dashed side box per thing beside the ladder (a borrowed floor such as optimistix's adjoints, the pipekit-cycle protocols, the setup functions that build its structures); copy `vardax_roles/`, and say in the PR which names were left off (protocols, exceptions, plotting, deprecated aliases, stubs).

## Dependencies

The library at the top with its icon and tagline, the stack layers as boxes below, one card per package that imports it (what it is, and the library's modules it uses in small mono), dashed for an import through an extra with the extra named under it, grey arrows between dependents, a strip saying what the library builds on, and a legend. The edges are geostack's `docs/api/dependency_graph.md`; the modules are each package's `from <pkg>… import` lines on `main`. Copy `argminax_ecosystem/`. When every dependent imports the library through an extra, draw the arrow to the library dashed too; when every name it uses comes from the package root, label each card with the areas it uses and the methods it calls, and keep the legend to the marks used (`vardax_ecosystem/`).

## Code map

The library's public modules as cards inside one package box: each card has the module's name, what it does in a few words, and the libraries it imports as chips (dashed when through an extra). Arrows run only between the library's own modules and read "imports"; a module that imports nothing else from the package says so under its card. A strip names what every module imports. Read the `import` lines of each `_src/<module>/`, including the lazy ones inside functions (an extra is usually imported there). Copy `argminax_imports/`. A library with many areas under `_src/` and one public root groups the areas into cards by job (each card lists its areas in mono), draws arrows only for imports between cards, routes an arrow between neighbours in a row underneath the row, and names the areas every card imports (types, errors, protocols, utilities) in a strip; copy `vardax_imports/`. An AST pass over `_src/` gives the edges, and each edge between cards is then checked by hand.

The exception hierarchy is a code map too: the standard-library parents dashed along the top, the package's base error in the middle, and each concrete error below with an arrow to each parent, what raises it in plain words, and where (the modules that raise it, or the extras it names). A strip says what is reported without an exception, such as a solve's `result`. Read the classes in the exceptions module and their `raise` sites; the figure follows the code where a docstring has fallen behind. Copy `argminax_exceptions/`.

## Hub

The central object in the middle lane, as one large equation with a short note on its shape and how it is held. A lane on the left for what builds it or uses it to get there, a lane on the right for what is read off it, three or four cards each. Every connector is labelled with the operation that links the card to the object (a solve, a factor, a log-determinant, a fixed point). Each card has a title in plain words, its module and public names, and its formula. Two slabs say what does the operations and what the large-scale variant is. Copy `argminax_hessian/`.

## Comparison

Two to four columns of the same height, one per way of doing the job, each drawn with the same picture so the difference is the only thing that changes (argminax: the chain of iterates, with the differentiated ones filled and the rest hollow). Under the picture, one equation and three rows of facts with the same keys in every column (argminax: adjoint, costs, right when). A key explains the marks; a slab says what all the columns share. Take the facts from the docstrings of the classes being compared. Copy `argminax_adjoints/`. `vardax_adjoints/` has four columns: the same rollout drawn in each, with filled states kept for the backward pass, and its facts from the page's own memory and time tables.

## Gallery

One column per structure or family, each with a pictogram of its shape (a small matrix drawn as cells, generated by a script before `ready()`), its defining equation, and the same rows of facts in each (storage, cost of a step, what it accepts). A full-width lane below for the special cases, and slabs for what reads every family out and what does its algebra. Take the facts from the table the gallery sits above, and keep its caveats (a cost that holds only for a built-in option). Copy `argminax_blr_families/`. A family that is a method rather than a shape gets a sketch of what it does to one small object instead: `vardax_posterior/` draws each posterior family in a two-dimensional state (the prior and one observation's likelihood band giving the exact posterior, a Gaussian matched to a curved cost at the MAP, the CG steps behind a mat-vec, an inflated ensemble ellipse), computed from 2×2 algebra in the script before `ready()`.

## Specimen

A row or a grid of panels, each the real output of one public call on toy data, drawn by matplotlib from the library's `docs/assets/readme/figures.py`. A basis specimen shows a few members chosen to show what changes along the family (degree, concentration, scale), each titled with its index and the number that ranks it; a map or projection shows the truth, the data, the estimate and its error, with the error's scale stated. The suptitle and subtitle say what to read off it. Keep each figure to one idea and six panels or fewer. Copy functax's `docs/assets/readme/figures.py`: its `run` executes a page's Python blocks, and its `mollweide` helper draws a field on the sphere.

## Module detail

The overview's four lanes for one module: how the caller drives it, its variants, its state, and what it returns, with slabs for the libraries it rests on. Copy `argminax_blr/`.

## Pipeline

An end-to-end example as four lanes joined by labelled `wire` connectors: the inputs of one event, the operators built from them, the method that runs, and what comes out and where it is handed on. Every box carries a status chip in the `code-sketch` markers (today, today*, your code, not pinned), read from the example's own sketch, and a not-pinned box has a dashed border. A band under the lanes shows the slot that varies between versions of the example (the forward model's tiers, the instruments), one card per option, each saying what fills the slot on its library's `main` and which of this library's names it calls; a dashed arrow runs from the band to that slot. A strip of steps places the example in the cycle it belongs to, with this example's step outlined, and a key explains the chips. Draw the page's code, not its old diagram, and check every name in the band against the other library's `__all__`. Copy `vardax_methane/`.
