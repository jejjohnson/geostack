---
name: library-math-reference
description: The GeoStack standard for a library's mathematical reference, the numbered chapters of its docs that derive what the code computes. How to plan the chapter list from the library's public names (foundation, methods or operations, cross-cutting, numerics, worked examples, notation, references), the section order of each kind of chapter, the real problem from remote sensing, geoscience or mathematical modelling that motivates each chapter and carries its explanation, and the MyST forms every chapter uses (numbered and labelled equations with a where-sentence of shapes, labelled code-block directives in the code-sketch format with their printed output, labelled tables, cite roles on a bibliography, xref links into the API), the notation page with its classic-letters table, the test that runs every block and checks what it prints, and the myst.yml wiring. vardax, filterax, integrix and functax are the examples it was taken from, not the standard itself, and the skill says where each still differs. Use when planning, writing, reviewing or extending a stack library's mathematical reference, when adding a chapter for a new method, or when bringing an existing reference up to the standard.
---

# Library mathematical reference

Draft 3 · 19 Nov 2026 · taken from the mathematical references of filterax (`docs/math/`, 20 chapters), vardax (`docs/`, 23 chapters) and integrix (`docs/theory/`, 13 chapters), and from functax's guide pages (`docs/guide/`), which set how a real problem motivates and carries the explanation.

A stack library's docs have a mathematical reference: numbered chapters that write down what the library computes, derive it, run it and say when it fails. This skill is the standard for those chapters. The three references it was taken from are examples, not the standard: each was written before this skill, each does some things the others do better, and §9 lists what each would change to meet it. Where an example and this skill disagree, follow the skill, and where the skill is silent, copy the example named for that rule.

It builds on five other skills and does not repeat them: `role-ladder` (in this plugin) for the roles the library's names play, which decide the chapter list and the order inside a chapter; `eman-style-math` for the notation; `code-sketch` for the code on the page; `eman-prose-tech` for the voice; and `library-docs-figures` (in this plugin) for any figure a chapter carries.

## 1. Planning the reference

1. **Put every public name on a rung.** Read `__all__` on `main` and ask the eight questions of `role-ladder` (§2 there) of each name. A name that seems to sit on two rungs is two things packaged together; split it before planning its chapter, because the chapter will have to explain them apart.
2. **Write the chapter list in five groups, from the rungs.** Foundation, methods (or operations, for a toolbox library), cross-cutting, numerics, worked examples. A reference page for notation and one for references close the section. All three examples use these groups in `myst.yml`.

   | Rung | Goes in | Example |
   |---|---|---|
   | ⓪ structure, ③ model | Foundation: what the library acts on and what it holds | vardax's observation-model and dynamical-model chapters |
   | ① primitive | Foundation, or Operations in a toolbox library | integrix's rules and scans |
   | ② strategy | Cross-cutting, one chapter per strategy that several methods share | filterax's localization and inflation, vardax's adjoints |
   | ④ algorithm | Methods, one chapter each | filterax's ETKF, vardax's 3DVar |
   | ⑤ step, ⑥ recipe | the chapter on the cycle, and the worked examples | filterax's assimilation cycle |

   Numerics has no rung of its own: it holds the exact methods the others are checked against, and what precision and gradients do to them. A library need not have every rung, so it need not have every group.
3. **Order the chapters so each needs only the ones before it.** A chapter cites earlier equations by label and never a later one, except in its "See also".
4. **One idea per chapter.** A chapter that needs two unrelated agreement checks is two chapters. The examples run from about 120 to 1,200 lines; a method chapter is usually 400 to 800.
5. **Propose the list before writing**, with one line per chapter saying what it derives and which public names it covers, then write one chapter at a time and show it before the next.

## 2. The kinds of chapter

Every chapter is written on a real problem (§3): it opens with one before any equation, and the opening ends with a short paragraph that says what the chapter covers, in order. A chapter may also say which limit of the previous chapter it removes, as filterax's `06_etkf.md` does, but that comes after the problem and does not replace it.

| Kind | Derives | Sections, in order | Copy from |
|---|---|---|---|
| Foundation | the object or problem the whole library is about, written once | the scenario; the definition; the forms it takes in practice; what the library does with each; See also | integrix `01_integral.md`, filterax `01_problem_setting.md` |
| Method | one algorithm or model | the update or cost; Why …, one section per idea that makes it work; Algorithm; Implementation in `<pkg>`; Agreement with …; Numerics; When it breaks; When `<method>` is the right answer; See also | filterax `06_etkf.md`, vardax `05_threedvar.md` |
| Cross-cutting | a strategy several methods share | the problem it solves; the idea; the variants, with a table of when and cost; Implementation in `<pkg>`; Numerics; When it breaks; When it is the right answer; See also | filterax `11_localization.md`, integrix `09_adjoints.md` |
| Numerics | exact reference methods, precision, gradients | what they compute; Algorithm; the library's methods against the reference; When it breaks; See also | filterax `17_exact_references.md`, integrix `11_precision.md` |
| Worked example | one end-to-end problem that uses most of the library | the setup; one run, end to end; the methods against each other; what the failures look like; When this setup is the right answer; See also | filterax `19_lorenz96_example.md`, integrix `13_methane_example.md` |

Section headings are sentences about the content ("Why the symmetric square root", "A scan's adjoint is a reversed scan"), not generic labels and not rung names, except the fixed ones in the table. Four of the fixed sections carry the standard's main promises:

- **Algorithm** writes the method out by hand in plain JAX, with the dense quantities the production code avoids, so a reader can match every line to an equation.
- **Implementation in `<pkg>`** calls the public API on a small stated problem and prints a result.
- **Agreement with …** checks the library against an exact answer (a closed form, a dense solve, the linear-Gaussian limit) and against the by-hand algorithm, and prints the differences.
- **When it breaks** demonstrates a failure with code and a printed number, then names the chapter that treats it. A list of other limits may follow, but one failure is always shown.

### The order inside a chapter

A method chapter explains in the order `role-ladder` gives (§3 there): nouns, verbs, adverbs, the noun phrase, then the sentence. The sections above already follow it, and the chapter keeps to it.

- **The opening equations name the structures**, each with its shape and, for a physical quantity, its unit.
- **The "Why …" sections name the primitives and the strategies**: what is called on those structures, and what a caller can swap.
- **Algorithm is the sentence.** It uses only pieces the chapter has already named.
- **Implementation is the step** that wraps the algorithm with its settings. It says in words what state the model holds between calls and what builds it.
- **A chapter that names more than three pieces carries the pieces table** (Piece, Rung, Kind, Used by) as a labelled `{table}`, before the Algorithm section.
- **Every strategy named is checked.** Naming a strategy claims that the maths does not change when it is swapped, so the Agreement section prints that check for each one, or says what does change. filterax's `06_etkf.md` does this for its two routes, which agree in float64 and part in float32.
- **Write rung names, not layer numbers**: "the analysis step" and "the algorithm it wraps", not "Layer 2" and "Layer 1".

### The floor, and who owns a derivation

A reference is written for readers who work at ② to ④ and treat the rung below as given. A chapter says in one line what that floor guarantees and links to the chapter of the library that owns it; it goes one rung down, never two. So a derivation lives in the reference of the library that owns the rung: filterax's ETKF chapter derives the gradient rule that gaussx installs, and under this rule gaussx's reference derives it and filterax cites it. Where the lower library has no chapter yet, the derivation stays where it is, with a note naming where it belongs.

## 3. Real problems

A chapter is motivated by a real problem and explained on it. The problem comes from remote sensing, the geosciences or mathematical modelling, it is one a reader of the stack would meet in their own work, and it stays on the page from the first paragraph to the last printed number. functax's guides and integrix's chapters are the examples; filterax's and vardax's method chapters mostly run on abstract states and are the ones to change (§9).

- **Open on the problem, in physical terms.** Say what is measured or modelled, in which units, and what goes wrong without the chapter's idea. integrix's `05_rules.md` opens on a global mean surface temperature that a plain average gets 5 K too cold; functax's `regridding.md` opens on a flux that loses mass at every coarsening.
- **Let the problem ask for each equation.** An equation enters because the problem needs it, and the sentence before it says so. The general form comes second: integrix's `01_integral.md` writes five integrals from one day of a methane satellite, then reads the domain, the measure and the integrand off each.
- **Run the code on the problem.** Variables carry the names of the physical quantities (`wavelength`, `band_areas`, `wind`), sizes are of a real kind (30 m pixels, a 2° grid, 20 observed sites of 40) even when their count is toy, and constants are physical and stated with units. A block that draws a random matrix with no meaning is not an example.
- **Read the printed numbers back in the problem's terms.** "The middle band loses about 4 % of its light", not "the second entry is 0.9602".
- **Declare the problem's symbols in a table on the page**, with shapes and units, as `01_integral.md` and functax's `methane.md` do.
- **Simulate the data and say where real data drops in.** The problem is real and the data is toy: a few lines build the scene, so the chapter runs anywhere, and one sentence names the product whose values would replace it ("the pixel centres and values of an L2 product drop in for `pixels` and `observations`").
- **Quote nothing about an instrument or a product that was not read.** A resolution, a noise level or a band is either cited or called a toy value.
- **Keep one problem per chapter where one will do**, and prefer a problem an earlier chapter already set up. An abstract case (a polynomial a rule integrates exactly, the linear-Gaussian limit) belongs in the Agreement section, after the real problem, because it is what has an exact answer.
- **The worked example uses the stack's running example where the library can.** functax's `methane.md` names it: one methane overpass, mapped over a 100 × 100 grid. integrix's `13_methane_example.md` takes a plume from spectrum to emission rate through every module.

Problems the examples already use, as a place to start:

| Field | Problem | In |
|---|---|---|
| Remote sensing | a band radiance under a spectral response; a column average of methane; a plume's emission rate from its mass or its flux across a transect; reading a grid at a swath's pixels | integrix `01_integral.md`, `13_methane_example.md`; functax `quadrature.md`, `methane.md` |
| Geoscience | a global mean on a latitude–longitude grid; ocean heat content; conservative regridding with a coastal mask; gridding stations | integrix `01_integral.md`, `05_rules.md`; functax `regridding.md`, `gridding.md` |
| Mathematical modelling | a Lorenz-96 twin experiment; 4D-Var on an inertial oscillation | filterax `19_lorenz96_example.md`; integrix `09_adjoints.md` |

## 4. Equations and notation

- **Every display equation is on one source line and labelled**: `$$ … $$ (eq-<chapter>-<name>)`. It is followed by a sentence starting "where" that gives each new symbol and its shape, and it is cited as `[](#eq-…)`. The frontmatter `numbering: enumerator: "N.%s"` numbers it with the chapter.
- **The notation is `eman-style-math`'s**, with the kind-named macros (`\vect`, `\mat`, `\gvect`, `\gmat`, `\op`) installed in `myst.yml`. A library that departs from it says so at the top of its notation page with the reason, as integrix does for $f$ (the integrand there, the dynamics elsewhere).
- **The notation page** has one labelled table per topic with the columns Symbol, Shape, Meaning and In code, where "In code" is the argument or attribute that holds the symbol. It ends with a "Classic letters" table that maps the literature's letters to the house ones, so a reader coming from a paper can find their $\mat{K}$. Physical symbols carry units in the Shape column. Copy from integrix `notation.md` for the tables and filterax `notation.md` for the classic letters.
- **A symbol used on one page only** is declared in a table on that page, not on the notation page.

## 5. Code on the page

Every block is a `code-sketch`: a `# Shapes:` header, the equation in Unicode above each line, a shape comment on each array, and a status marker (`# today (<pkg>)`, `# user`) above each call into the stack or each function the reader would write.

````markdown
```{code-block} python
:caption: One analysis step, which tightens the observed components and leaves the others nearly alone.
:label: code-etkf-analysis
…
print(f"{result.log_likelihood:.3f}")
```

```text
-3.401
```
````

- **A caption and a label on every block.** The caption says what the block shows, as a sentence; the label is `code-<chapter>-<name>` and the prose cites it as `[](#code-…)`.
- **The output follows in a `text` fence**, exactly as printed. The prose after it reads the numbers and says what they mean; it never quotes a number that no block prints.
- **A chapter's blocks share one namespace**, top to bottom, so later blocks reuse earlier names and imports. A chapter that needs float64 switches it on in its first block.
- **Toy sizes.** A chapter runs in under about half a minute on a laptop CPU.
- **A block that cannot run** (a sketch with placeholders) starts with `# not run` and has no output fence.

## 6. The test

Each library has one test, `tests/test_math_pages.py`, in the slow tier: it runs every chapter's blocks in one namespace per chapter, each chapter in its own interpreter, and asserts that each block prints exactly the `text` fence that follows it. integrix's `tests/test_theory_pages.py` is the model; its pattern is extended to match the `{code-block}` directive with its option lines, as filterax's `tests/test_theory_code.py` does. A chapter is not finished until this test passes on it.

## 7. The MyST forms

The reference is written for mystmd and uses its directives and roles wherever one exists, so that everything a reader might cite has a label and everything that can be checked is checked by a strict build.

| For | Write | Not |
|---|---|---|
| An equation | `$$ … $$ (eq-…)` and `[](#eq-…)` | "equation 3 above" |
| Code | `{code-block} python` with `:caption:` and `:label:` | a bare fence |
| A table the text cites | `:::{table} <caption>` with `:label: tbl-…` | a bare table |
| A citation | `` {cite:p}`key` `` or `` {cite:t}`key` ``, with the entry in `docs/bib/*.bib` | "(Higham, 2002)" in the text, or a hand-written list |
| A public name | `` [`weighted_mean`](xref:api#<pkg>.<name>) ``, by the top-level name | an absolute URL into the API site |
| Another chapter | `[chapter 5](05_stochastic_enkf.md)` | "chapter 5" as plain text |
| A figure | `{figure}` with `:alt:` and `:label: fig-…`, drawn as `library-docs-figures` says | a mermaid block or an image in the repo |

A bibliography entry is added only from a source that was read: the authors, year, title and venue are checked against the paper or its publisher's page, and a DOI is included only when it resolves. The references page is the generated bibliography, grouped by the `.bib` file each entry is in.

## 8. Wiring

- **Folder.** `docs/math/`, with chapters `NN_slug.md`, `notation.md` and `references.md`.
- **`myst.yml`.** The `math:` macros, `bibliography:` listing the `.bib` files, `references.api` for the API inventory, and a toc entry titled "Mathematical reference" with one child per group and "Reference" last. Copy filterax's.
- **Line breaks.** One sentence per source line in a new reference; in an existing one, whatever its pages already use. Never a hard wrap at a column.
- **Build.** `myst build --html --strict` with the latest mystmd, so an unresolved label, citation or `xref:` fails.
- **Landing page.** `index.md` links the first chapter with one sentence on what the reference covers.

## 9. Bringing an existing reference to the standard

The three examples predate this skill. Each change below is a docs-only PR in the library, one group of chapters at a time.

| Library | Already meets | To change |
|---|---|---|
| filterax | folder, section order, labelled code blocks, cite roles and `.bib` files, one sentence per line, a test that runs every block | outputs move from "It prints …" sentences into `text` fences and the test checks them; API links become `xref:`; cited tables get labels; the hand-written `references.md` gives way to the bibliography; method chapters move from abstract states to a real problem that runs through the chapter; "Layer 1" and "Layer 2" become rung names; the derivations of what gaussx computes move to gaussx's reference once it has the chapter |
| vardax | section order of the method chapters, labelled code blocks, the notation page | chapters move to `docs/math/`; hard-wrapped paragraphs are unwrapped; per-chapter reference lists become cite roles on `.bib` files; outputs move into `text` fences; a `tests/test_math_pages.py` is added; chapter mentions become links; decision numbers leave the chapters; method chapters move from abstract states to a real problem |
| integrix | real problems throughout, `text` output fences checked by the test, labelled tables, `xref:` links, units | the folder is renamed from `docs/theory/` to `docs/math/`; code blocks get captions and labels; author-and-year mentions become cite roles on `.bib` files; method and cross-cutting chapters gain "When it breaks" and "See also" where they lack them |

functax has guide pages and no mathematical reference yet; its reference is planned from §1, and its guides keep their job of showing how to call the library.

A move of folder changes every chapter's URL, so the PR that does it lists the old and new paths and the pages that link in from outside the section.

## Checklist

- [ ] Every public name is on one rung, and the chapter list was proposed and agreed before any chapter was written, in the five groups, each chapter needing only those before it.
- [ ] Each chapter opens on a real problem in physical terms, with a paragraph that says what it covers; the problem asks for each equation, the code runs on it with physical names and units, and the printed numbers are read back in its terms.
- [ ] The data is simulated, the chapter says where real data drops in, and no fact about an instrument or product is uncited.
- [ ] Each chapter has the sections of its kind, in order; a method chapter has Algorithm, Implementation, Agreement and When it breaks, the last with a demonstrated failure.
- [ ] A method chapter names structures, then primitives and strategies, then the model, then the algorithm; it carries the pieces table when it names more than three pieces, and checks every strategy it names.
- [ ] A chapter goes one rung below its own at most, and links to the owning library's chapter for anything lower.
- [ ] Every display equation is on one line, labelled, and followed by a where-sentence with shapes; every symbol is on the notation page or in a table on its own page.
- [ ] Every code block is a labelled, captioned `code-sketch` followed by its exact output; no number in the prose lacks a block that prints it.
- [ ] `tests/test_math_pages.py` passes on the chapter.
- [ ] Citations are cite roles on checked `.bib` entries; public names are `xref:` links; cited tables and figures are labelled.
- [ ] The strict build passes with the latest mystmd.
