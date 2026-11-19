---
name: library-math-reference
description: The GeoStack standard for a library's mathematical reference, the numbered chapters of its docs that derive what the code computes. How to plan the chapter list from the library's public names (foundation, methods or operations, cross-cutting, numerics, worked examples, notation, references), the section order of each kind of chapter, and the MyST forms every chapter uses (numbered and labelled equations with a where-sentence of shapes, labelled code-block directives in the code-sketch format with their printed output, labelled tables, cite roles on a bibliography, xref links into the API), the notation page with its classic-letters table, the test that runs every block and checks what it prints, and the myst.yml wiring. vardax, filterax and integrix are the examples it was taken from, not the standard itself, and the skill says where each still differs. Use when planning, writing, reviewing or extending a stack library's mathematical reference, when adding a chapter for a new method, or when bringing an existing reference up to the standard.
---

# Library mathematical reference

Draft 1 · 19 Nov 2026 · taken from the mathematical references of filterax (`docs/math/`, 20 chapters), vardax (`docs/`, 23 chapters) and integrix (`docs/theory/`, 13 chapters).

A stack library's docs have a mathematical reference: numbered chapters that write down what the library computes, derive it, run it and say when it fails. This skill is the standard for those chapters. The three references it was taken from are examples, not the standard: each was written before this skill, each does some things the others do better, and §8 lists what each would change to meet it. Where an example and this skill disagree, follow the skill, and where the skill is silent, copy the example named for that rule.

It builds on four other skills and does not repeat them: `eman-style-math` for the notation, `code-sketch` for the code on the page, `eman-prose-tech` for the voice, and `library-docs-figures` (in this plugin) for any figure a chapter carries.

## 1. Planning the reference

1. **List what the library computes.** Read `__all__` on `main` and group the public names by the job they do, using the rungs of `role-ladder`: each algorithm or model is a candidate method chapter, and each strategy that several methods share (an adjoint, a localization, a rule for precision) is a candidate cross-cutting chapter.
2. **Write the chapter list in five groups.** Foundation, methods (or operations, for a toolbox library), cross-cutting, numerics, worked examples. A reference page for notation and one for references close the section. All three examples use these groups in `myst.yml`.
3. **Order the chapters so each needs only the ones before it.** A chapter cites earlier equations by label and never a later one, except in its "See also".
4. **One idea per chapter.** A chapter that needs two unrelated agreement checks is two chapters. The examples run from about 120 to 1,200 lines; a method chapter is usually 400 to 800.
5. **Propose the list before writing**, with one line per chapter saying what it derives and which public names it covers, then write one chapter at a time and show it before the next.

## 2. The kinds of chapter

Every chapter opens with a concrete problem before any equation: a scenario from geoscience (integrix's `01_integral.md` follows one day of a methane satellite) or the limit of the previous chapter that this one removes (filterax's `06_etkf.md`). The opening ends with a short paragraph that says what the chapter covers, in order.

| Kind | Derives | Sections, in order | Copy from |
|---|---|---|---|
| Foundation | the object or problem the whole library is about, written once | the scenario; the definition; the forms it takes in practice; what the library does with each; See also | integrix `01_integral.md`, filterax `01_problem_setting.md` |
| Method | one algorithm or model | the update or cost; Why …, one section per idea that makes it work; Algorithm; Implementation in `<pkg>`; Agreement with …; Numerics; When it breaks; When `<method>` is the right answer; See also | filterax `06_etkf.md`, vardax `05_threedvar.md` |
| Cross-cutting | a strategy several methods share | the problem it solves; the idea; the variants, with a table of when and cost; Implementation in `<pkg>`; Numerics; When it breaks; When it is the right answer; See also | filterax `11_localization.md`, integrix `09_adjoints.md` |
| Numerics | exact reference methods, precision, gradients | what they compute; Algorithm; the library's methods against the reference; When it breaks; See also | filterax `17_exact_references.md`, integrix `11_precision.md` |
| Worked example | one end-to-end problem that uses most of the library | the setup; one run, end to end; the methods against each other; what the failures look like; When this setup is the right answer; See also | filterax `19_lorenz96_example.md`, integrix `13_methane_example.md` |

Section headings are sentences about the content ("Why the symmetric square root", "A scan's adjoint is a reversed scan"), not generic labels, except the fixed ones in the table. Four of the fixed sections carry the standard's main promises:

- **Algorithm** writes the method out by hand in plain JAX, with the dense quantities the production code avoids, so a reader can match every line to an equation.
- **Implementation in `<pkg>`** calls the public API on a small stated problem and prints a result.
- **Agreement with …** checks the library against an exact answer (a closed form, a dense solve, the linear-Gaussian limit) and against the by-hand algorithm, and prints the differences.
- **When it breaks** demonstrates a failure with code and a printed number, then names the chapter that treats it. A list of other limits may follow, but one failure is always shown.

## 3. Equations and notation

- **Every display equation is on one source line and labelled**: `$$ … $$ (eq-<chapter>-<name>)`. It is followed by a sentence starting "where" that gives each new symbol and its shape, and it is cited as `[](#eq-…)`. The frontmatter `numbering: enumerator: "N.%s"` numbers it with the chapter.
- **The notation is `eman-style-math`'s**, with the kind-named macros (`\vect`, `\mat`, `\gvect`, `\gmat`, `\op`) installed in `myst.yml`. A library that departs from it says so at the top of its notation page with the reason, as integrix does for $f$ (the integrand there, the dynamics elsewhere).
- **The notation page** has one labelled table per topic with the columns Symbol, Shape, Meaning and In code, where "In code" is the argument or attribute that holds the symbol. It ends with a "Classic letters" table that maps the literature's letters to the house ones, so a reader coming from a paper can find their $\mat{K}$. Physical symbols carry units in the Shape column. Copy from integrix `notation.md` for the tables and filterax `notation.md` for the classic letters.
- **A symbol used on one page only** is declared in a table on that page, not on the notation page.

## 4. Code on the page

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

## 5. The test

Each library has one test, `tests/test_math_pages.py`, in the slow tier: it runs every chapter's blocks in one namespace per chapter, each chapter in its own interpreter, and asserts that each block prints exactly the `text` fence that follows it. integrix's `tests/test_theory_pages.py` is the model; its pattern is extended to match the `{code-block}` directive with its option lines, as filterax's `tests/test_theory_code.py` does. A chapter is not finished until this test passes on it.

## 6. The MyST forms

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

## 7. Wiring

- **Folder.** `docs/math/`, with chapters `NN_slug.md`, `notation.md` and `references.md`.
- **`myst.yml`.** The `math:` macros, `bibliography:` listing the `.bib` files, `references.api` for the API inventory, and a toc entry titled "Mathematical reference" with one child per group and "Reference" last. Copy filterax's.
- **Line breaks.** One sentence per source line in a new reference; in an existing one, whatever its pages already use. Never a hard wrap at a column.
- **Build.** `myst build --html --strict` with the latest mystmd, so an unresolved label, citation or `xref:` fails.
- **Landing page.** `index.md` links the first chapter with one sentence on what the reference covers.

## 8. Bringing an existing reference to the standard

The three examples predate this skill. Each change below is a docs-only PR in the library, one group of chapters at a time.

| Library | Already meets | To change |
|---|---|---|
| filterax | folder, section order, labelled code blocks, cite roles and `.bib` files, one sentence per line, a test that runs every block | outputs move from "It prints …" sentences into `text` fences and the test checks them; API links become `xref:`; cited tables get labels; the hand-written `references.md` gives way to the bibliography; openings get a concrete scenario where they have none |
| vardax | section order of the method chapters, labelled code blocks, the notation page | chapters move to `docs/math/`; hard-wrapped paragraphs are unwrapped; per-chapter reference lists become cite roles on `.bib` files; outputs move into `text` fences; a `tests/test_math_pages.py` is added; chapter mentions become links; decision numbers leave the chapters |
| integrix | scenario openings, `text` output fences checked by the test, labelled tables, `xref:` links, units | the folder is renamed from `docs/theory/` to `docs/math/`; code blocks get captions and labels; author-and-year mentions become cite roles on `.bib` files; method and cross-cutting chapters gain "When it breaks" and "See also" where they lack them |

A move of folder changes every chapter's URL, so the PR that does it lists the old and new paths and the pages that link in from outside the section.

## Checklist

- [ ] The chapter list was proposed and agreed before any chapter was written, in the five groups, each chapter needing only those before it.
- [ ] Each chapter opens with a concrete problem and a paragraph that says what it covers.
- [ ] Each chapter has the sections of its kind, in order; a method chapter has Algorithm, Implementation, Agreement and When it breaks, the last with a demonstrated failure.
- [ ] Every display equation is on one line, labelled, and followed by a where-sentence with shapes; every symbol is on the notation page or in a table on its own page.
- [ ] Every code block is a labelled, captioned `code-sketch` followed by its exact output; no number in the prose lacks a block that prints it.
- [ ] `tests/test_math_pages.py` passes on the chapter.
- [ ] Citations are cite roles on checked `.bib` entries; public names are `xref:` links; cited tables and figures are labelled.
- [ ] The strict build passes with the latest mystmd.
