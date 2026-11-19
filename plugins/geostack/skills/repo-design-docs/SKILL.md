---
name: repo-design-docs
description: The GeoStack standard for a library's design docs, the pages written before the code that say what the library is, what it owns, how it is built and why. The set (a README with the reading order, vision, boundaries, architecture, a decision log, API contracts by rung of the role ladder, and examples, with optional feature deep dives, integration recipes and research notes), what each page holds, the form of a decision record, how the docs are written before implementation and turned into issues, and how they are kept honest once code exists. filterax, vardax, xtremax and xr_toolz are the examples it was taken from, not the standard itself, and the skill says where each differs. Use when starting a library or a large feature, when writing or reviewing a design page, a boundary or a decision record, when a design and the code have drifted, or when bringing an existing design-doc set up to the standard.
---

# Library design docs

Draft 2 · 19 Nov 2026 · taken from the design docs of filterax (`docs/design_docs/`), vardax (`docs/design/`), xtremax (`docs/design_docs/`) and xr_toolz (`docs/design/`), which share one layout, and from the conventions of geostack's own design pages (`docs/overview.md`, "How the pages are written").

A stack library is designed on the page before it is built. Its design docs say what the library is and is not, what it owns, how its pieces fit, and why each contested choice went the way it did. The four sets this skill was taken from are examples, not the standard: they agree on the layout, and §8 lists where each differs from what is written here.

It builds on other skills of this plugin and does not repeat them: `role-ladder` for the rungs the API is organised by, `style-math` for the notation, `style-code-sketch` for every interface sketch, and `six-step-cycle`, `data-levels` and `modeling-cycle` for saying where the library sits. The design docs are one of three kinds of page in a library's docs, and each has its own job:

| Page | Says | Standard |
|---|---|---|
| Design docs | what will be built, what it owns, and why | this skill |
| Mathematical reference | what the code computes, derived and run | `repo-math-reference` |
| README | what the library is, in one screen | `repo-readme` |

## 1. The set

The folder is `docs/design/`. Every page carries `status:` (draft, accepted or superseded) and `version:` in its frontmatter.

| File | Holds | Copy from |
|---|---|---|
| `README.md` | one line on what the library is, the tree of the folder with one comment per file, and the reading order | filterax `README.md` |
| `vision.md` | why the library exists (§2) | xtremax `vision.md`, vardax `vision.md` |
| `boundaries.md` | what it owns and what it hands off (§3) | vardax `boundaries.md` |
| `architecture.md` | how it is built (§4) | filterax `architecture.md` |
| `decisions.md` | the decision log (§5) | filterax `decisions.md` |
| `api/` | the contracts, one page per band of the ladder (§6) | filterax `api/` |
| `examples/` | how the contracts are used, on the same bands, plus integration | filterax `examples/` |

Three more folders appear only when the library needs them:

| Folder | Holds | Example |
|---|---|---|
| `features/` | a deep dive per family of methods: its scope, the maths the family shares, one entry per method to build, the infrastructure they share and a comparison table | filterax `features/filters.md` |
| `integrations/` | one recipe per library it meets, with which side owns what | filterax `integrations/pipekit.md` |
| `research/` | background theory and audits of existing code that the design draws on | filterax `research/` |

The reading order is vision, architecture, boundaries, the API pages from the bottom band up, the feature pages, the examples, the integrations, and the decisions last.

## 2. Vision

- **A one-liner**, then the motivation: the problem as its users meet it, on a real case.
- **User stories**, each naming who and what they need.
- **The single equation**, where the library has one that every method specialises (vardax's cost; `repo-readme` §3 asks the same of a README).
- **Design principles**, few enough to be checked against.
- **Identity**: an "is" list and an "is not" list. The second does more work, because it is what stops scope from growing.
- **Where it sits**: the level transition it computes (`data-levels`), the steps of the `six-step-cycle` it serves, and the libraries above and below it.

## 3. Boundaries

- **What the library does not do**, each item with the library that does it instead.
- **The ownership map**: a table of concern and owner, the library's own concerns included.
- **The "where does X go" test**: a table of questions, each with the library that answers yes. A reader with a new piece of code should be able to place it from this table alone (vardax's is the model).
- **Ecosystem interactions**: one short sketch per neighbour, showing the hand-off.
- **The dependency graph**, with what is required, what is behind an extra, and what only examples use.
- **Scope**: in and out, for the first release.
- **The testing strategy and the roadmap**, and the open questions that are not yet decisions.

The layers and the import rules of the stack are in geostack's `AGENTS.md` and its dependency graph; a library's boundaries page applies them and does not restate them.

## 4. Architecture

- **The rungs the library has**, in the words of `role-ladder`: its structures, primitives, strategies, models, algorithms and steps. A library need not have every rung.
- **The protocols**: each abstract Module, what it must provide, and which names implement it.
- **The data types** that cross its boundary, with shapes.
- **The package layout**, as a directory tree.
- **The dependencies**: required, optional, examples only.

## 5. Decisions

`decisions.md` is a log. Each record has an ID (`D1`, `D2`, …) that the other pages cite, and never changes its number.

```markdown
## D4: Particles-only state, with statistics computed on demand

**Status:** Accepted

**Context:** <the situation that forces a choice, and who is affected>

**Options:**
- (A) <one option, with what it costs>
- (B) <another>

**Decision:** <the option taken, and the reason in one or two sentences>

**Consequences:**
- <what follows, the costs included>
```

- **A decision is recorded when there were real options.** A choice nobody would contest is prose on the page it belongs to.
- **A record is never deleted or rewritten.** When it is overturned, its status becomes "Superseded by D12" and D12 says what changed.
- **An older record is brought to this form without adding to it.** Keep its ID, its title and what it decided. Fill a field only from what the source says; where the source gives no status, names no alternative or lists no consequence, write that the field is implied or not recorded, in the record itself. A rewrite that supplies a confident "Accepted" or a plausible rejected option has invented history.
- **Decision IDs stay in the design docs.** The mathematical reference and the code do not cite them (`repo-math-reference` §9).

## 6. API contracts and examples

`api/README.md` is the surface inventory: every planned public name, grouped by band, with one line each. The detail pages follow the bands of the ladder, bottom up, and `examples/` mirrors them:

| Page | Rungs | Each entry gives |
|---|---|---|
| `api/primitives.md` | ⓪ structures, ① primitives | the mathematical definition, the cost, and the signature |
| `api/components.md` | ② strategies, ③ models, and the protocols | what it holds, what it must provide, the signature |
| `api/models.md` | ④ algorithms, ⑤ steps | the method in one equation, what it composes, the signature |

- **Every signature is a `style-code-sketch`**, with jaxtyping dims and a status marker. A name that does not ship yet is `# proposed (D7)` or `# proposed (#123)`, naming the decision or the issue that adds it; a name that ships is `# today (<pkg>)`, checked in the repo.
- **Notation is `style-math`'s**, from the first draft. A design written in the literature's letters has to be rewritten when the code and the mathematical reference arrive.
- **An example composes what the API page declares** and nothing else. One that needs a name the API page lacks has found a gap in the design.

## 7. Writing them, and keeping them honest

**Before the code.** Write in this order, and get each agreed before the next: vision, boundaries, architecture with the decisions it forces, then the API pages and the examples. The backlog comes from the agreed design, as GitHub issues (the `report` and `gh-issue` skills; geostack's `docs/templates/wave-backlog.md` drafts a whole wave), not as pages in the repo.

**After the code.** The code wins. A design page that describes what exists is checked against it, as geostack's own pages are:

- **Every `proposed` marker becomes `today`** when the name ships, and the page's `version:` moves.
- **A "Differs" box marks each place where the code disagrees with the design**, with which of the two is right. The gap that matters becomes an issue; the design is corrected where the code is right.
- **The maths moves out.** Once a method ships, its derivation lives in the mathematical reference, and the feature page links to the chapter instead of keeping a second copy.
- **Every number on a page comes from code that asserts it.**

**Reviewing a set.** Start with whether the docs describe the library that exists, and only then look at their form.

1. **Read what the repository says about its own docs.** Its `AGENTS.md` or README may already say that the design pages are a draft or a roadmap, and which wins where they disagree with the code.
2. **Compare the docs with the package.** List the modules and the public names the pages describe and the ones the code has. What is documented and missing, and what exists and is undesigned, are the first findings: xtremax's pages describe a model zoo, max-stable processes and plotting that were never built, and barely mention the point-process package that was.
3. **Check the decisions against the code.** A record the code has overturned is superseded, and a choice the code plainly made with no record needs one.
4. **Then check the form** against §1 to §6.

Write the review most important first, say what can stay, and separate what one page's PR can fix from what needs a decision first (cut the unbuilt scope, or keep it as `proposed`).

## 8. Bringing an existing set to the standard

| Library | Already meets | To change |
|---|---|---|
| filterax | the full set with `features/`, `integrations/` and `research/`; decision records with status, context, options, decision and consequences | the folder is `docs/design_docs/`; pages still carry the library's earlier names (ekalmX, filterX) in their titles; "Layer 0, 1, 2" becomes the rungs; the API pages use the literature's letters and spacing macros; the README's count of decisions is stale; the wave backlog is kept as pages under `issues/` |
| vardax | the set, with the "where does X go" test and the list of what it does not do | paragraphs are hard-wrapped; "three-layer stack" becomes the rungs; decision records are prose without the five fields; extra pages (`latent_da.md`, `posterior.md`, `pipekit_composition.md`) sit at the top level where `features/` and `integrations/` would hold them |
| xtremax | the core set of pages | the pages describe a library that was not built (a model zoo, max-stable processes, GP, plotting and dataset modules) and leave out the point-process package that was, so the first change is to decide what is cut and what stays as `proposed`; its `AGENTS.md` already calls them a draft roadmap; the folder is `docs/design_docs/`; the README is `design_overview.md`; files repeat their folder's name (`api/api_primitives.md`); "Layer 0, 1, 2" becomes the rungs |
| xr_toolz | the core set, with extra folders of its own | pages for single features sit at the top level; the layout of its `bridges/` and `conventions/` folders is its own and may stay |

A move of folder changes every page's URL, so the PR that does it lists the old and new paths.

## Checklist

- [ ] The set is complete: README with reading order, vision, boundaries, architecture, decisions, `api/` and `examples/`; every page has `status:` and `version:`.
- [ ] The vision has an "is not" list, and says where the library sits on the levels and the six-step cycle.
- [ ] The boundaries page has the ownership map and a "where does X go" test that places new code.
- [ ] The architecture names rungs of the ladder, not layers.
- [ ] Each decision has an ID, a status, the context, the options, the decision and the consequences; none was deleted.
- [ ] Every signature is a code sketch in the house notation, marked `today` or `proposed` with the decision or issue that adds it.
- [ ] Each page was agreed before the next was written, and the backlog is issues.
- [ ] An older record brought to the standard form marks every field its source does not supply.
- [ ] Where code exists, the pages were checked against it: no stale `proposed`, a "Differs" box where they disagree, derivations in the mathematical reference.
