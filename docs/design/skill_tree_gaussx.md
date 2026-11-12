---
title: "gaussx's skill shelf: 24 use skills by role, seven cards, three listed"
subtitle: "Design · the skill tree worked out for one library"
short_title: Skill tree · gaussx
description: "Proposes gaussx's shelf of agent skills as the worked example of the skill tree: a shared role-ladder skill at the root, a shelf of 24 use skills tagged by role, grouped into seven units that each carry a contract card, three of them listed, the four existing contributor skills linked to the rungs they extend, and the third-party contracts each unit relies on. Measured on gaussx at 7fa1a38 and on the libraries that depend on it."
date: 2026-11-12
keywords:
  - agents
  - skills
  - gaussx
  - role ladder
  - skill tree
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

(sg-decision)=
## Decision

Recommended: gaussx's shelf is 24 use skills and four extend skills, each tagged with its role; they group into seven units that each carry a contract card; three skills are listed, one per band; and every skill builds on the shared `role-ladder` skill instead of explaining the ladder itself.
Alternative: seven listed skills, one per unit, which trigger more precisely and cost about 1.2k tokens in every session (estimate).
Status: open, 12 Nov 2026.

What does not change: gaussx's public API, the content of its four contributor skills, its maintenance skills, and the name of today's skill, which stays as the front door.

(sg-question)=
## Question

gaussx has one use skill for its whole API.
How should it split so that each library above gaussx gets only the part it uses, and what is the pattern the other 22 repositories can follow?

(sg-motivation)=
## Motivation

One skill covers a ladder of five rungs, and no dependent uses all of it.

gaussx @ `7fa1a38` exports 352 names in `gaussx.__all__` without the numpyro extra, and seven more with it.
Its one skill, `structured-gaussian-linalg`, is 156 lines and costs about 2.1k tokens to load.
Every count on this page comes from `scripts/skill_shelf.py`, which reads the submodules at the pins of this commit; `tests/test_skill_shelf.py` fails when a table here is stale.
[](#tbl-sg-units) counts the public names by unit, from the subpackage each one is imported from in `src/gaussx/__init__.py`.

:::{table} gaussx's public names by unit. A unit groups subpackages that the same dependents use; its rungs follow `AGENTS.md` @ `7fa1a38` and the ladder of the [kinds-and-roles](./kinds_and_roles.md) page, and its band is that page's.
:label: tbl-sg-units

<!-- shelf:units:begin -->
| Unit | Rungs | Home band | Subpackages | Names |
|---|---|---|---|---|
| Structures and primitives | ⓪ ① | numerics | `gaussx._operators` 38, `gaussx._primitives` 29, `gaussx._tags` 24, `gaussx._linalg` 20, `gaussx._gmrf` 12, `gaussx._sparse` 4 | 127 |
| Strategies | ②, setup | numerics, ML | `gaussx._strategies` 20, `gaussx._randomized` 10, `gaussx._sketching` 7, `gaussx._preconditioners` 5, `gaussx._solve_frontend` 2 | 44 |
| Gaussians | ③ | ML | `gaussx._expfam` 16, `gaussx._distributions` 11 | 27 |
| Gaussian processes | ④ | ML | `gaussx._gp` 29 | 29 |
| State space | ④ | ML | `gaussx._ssm` 53 | 53 |
| Quadrature | ② to ④ | ML | `gaussx._quadrature` 44 | 44 |
| Inference and ensembles | ④ | ML | `gaussx._inference` 28 | 28 |
| **All** | | | | **352** |
<!-- shelf:units:end -->
:::

[](#tbl-sg-consumers) counts the distinct gaussx names each dependent package takes, from its imports and its attribute accesses on `gaussx`.

:::{table} Which part of gaussx each dependent package uses. A package that declares gaussx and takes no public name from it is not listed.
:label: tbl-sg-consumers

<!-- shelf:uses:begin -->
| Dependent package | Names used | Structures and primitives | Strategies | Gaussians | Gaussian processes | State space | Quadrature | Inference and ensembles |
|---|---|---|---|---|---|---|---|---|
| pyrox-gp | 51 | 9 | 6 | 3 | 9 | 14 | 6 | 4 |
| kernellib | 25 | 16 | 9 | 0 | 0 | 0 | 0 | 0 |
| pyrox-lgm | 23 | 16 | 0 | 0 | 0 | 0 | 7 | 0 |
| argminax | 17 | 9 | 6 | 0 | 0 | 0 | 0 | 2 |
| filterax | 17 | 3 | 1 | 1 | 0 | 1 | 0 | 11 |
| plumax | 12 | 8 | 1 | 2 | 0 | 1 | 0 | 0 |
| gauss-flows | 7 | 0 | 0 | 0 | 0 | 4 | 2 | 1 |
| radtrax | 7 | 5 | 1 | 1 | 0 | 0 | 0 | 0 |
| spectraldiffx | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| optax-bayes | 4 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| manipy-jax | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| vardax | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 1 |
| finitevolx | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
<!-- shelf:uses:end -->
:::

Three groups stand out.
kernellib, spectraldiffx, optax-bayes and manipy stay at structures, primitives and strategies.
filterax and gauss-flows mostly call algorithms: inference and ensembles for one, state space for the other.
pyrox-gp uses every unit, and pyrox-lgm only structures, primitives and quadrature.

(sg-example)=
## Running example: two callers

The same two tasks are followed through every option.

**TL;DR.** An agent writing kernel ridge regression needs to know how to pass a solver; an agent writing a Poisson GP needs the integrator interface. Neither needs the other's part of gaussx.

1. **kernellib, kernel ridge regression at N = 10⁵.** The agent builds `K + σ²I` as an operator and solves it by preconditioned conjugate gradients. It needs the operator rules (⓪, ①) and the strategies (②).
2. **pyrox-gp, a sparse GP with a Poisson likelihood.** The agent needs an expected log-likelihood under a Gaussian, so it needs an integrator and a likelihood (quadrature), the Gaussian helpers (③) and a `solver=` to pass down.

:::{table} What each caller's skill carries about gaussx.
:label: tbl-sg-example

| Caller | Today | Proposed |
| --- | --- | --- |
| kernellib | prose mentions; to learn more it loads the whole gaussx skill, about 2.1k tokens | two cards (structures and primitives; strategies), at most 30 lines; it loads `solvers-and-preconditioners` only to tune the preconditioner |
| pyrox-gp | prose mentions; the whole gaussx skill | cards for the units it uses; it loads `gaussians-and-recipes` and its quadrature reference only to write a new likelihood |
:::

(sg-shelf)=
## The shelf

The shelf has three layers over a shared root: the catalogue of skills by role, the units that group them into cards, and the few that are listed.

```{mermaid}
flowchart TB
  subgraph dependents["Dependents"]
    direction LR
    pyrox
    filterax
    argminax
    kernellib
    radtrax
  end
  subgraph extend["gaussx · extend skills"]
    direction LR
    addrecipe["add-recipe"]
    addstrategy["add-solver-strategy"]
    addoperator["add-operator"]
    adddispatch["add-dispatch-path"]
  end
  subgraph use["gaussx · use skills"]
    direction LR
    recipes["gaussians-and-recipes<br/>③ ④"]
    solvers["solvers-and-preconditioners<br/>②"]
    front["structured-gaussian-linalg<br/>⓪ ① · front door"]
  end
  subgraph base["Shared bases"]
    direction LR
    roles["role-ladder"]
    lineax
    equinox
    numpyro
  end
  pyrox --> recipes
  pyrox --> solvers
  filterax --> recipes
  argminax --> solvers
  argminax --> recipes
  kernellib --> solvers
  radtrax --> front
  addrecipe --> recipes
  addstrategy --> solvers
  addoperator --> front
  adddispatch --> front
  recipes --> solvers
  recipes --> front
  solvers --> front
  recipes -.-> numpyro
  front --> lineax
  front --> equinox
  front --> roles
```

Each arrow reads "builds on", and an arrow that a longer path implies is left out.

### The root: the role-ladder skill

The ladder is explained once, in a listed skill of its own, and no library's skill repeats it.

`role-ladder` is the agent-facing form of the [kinds-and-roles](./kinds_and_roles.md) page.
It has a job outside the tree as well: it explains an algorithm or a library by the roles its pieces play, which is what an explainer page needs, so it is listed and can be asked for directly.
Every plugin in the stack depends on it, so it is always installed and its description is listed once.
A first draft exists in the test repository, as a personal skill; its home in the stack is geostack's plugin.

| Part | Holds |
| --- | --- |
| `SKILL.md` | the seven rungs with their grammar, "which rung is it?", how to explain an algorithm up the ladder, the bands and their floors |
| `references/structure.md` to `references/recipe.md` | one file per role, ⓪ to ⑥: what it is, how to tell it apart, how to explain it, how it goes wrong |
| `references/orchestration.md` | what runs recipes, and why it is not a rung |
| `references/kinds.md` | the three kinds (function, Module, Operator), what each may take, and the three invariants |
| `references/worked-example.md` | one job climbed up the whole ladder: piece, rung, kind, used by |

A unit builds on the role files of its rungs: gaussx's strategies unit on `strategy.md`, its Gaussians on `model.md`.
Three things then leave gaussx's own skills.
A skill says "⓪ structures" and does not define the word.
The generic JAX rules in today's "The rules your code must keep" move to the kinds file, and gaussx keeps only the rules that are its own.
The extend skills stop restating what a Module is.

### The shelf: 24 use skills by role

The shelf is the library: one short skill file per job a caller comes with, tagged with its role, so each can be pointed at on its own.
With `role-ladder` as the root, an entry never explains what a strategy or a model is, which keeps it short.

:::{table} gaussx's shelf. Each use skill is one file; the extend skills exist today in `.claude/skills/`. Names are from `gaussx.__all__` @ `7fa1a38`.
:label: tbl-sg-shelf

| Role | Skill | Covers | Subpackage |
| --- | --- | --- | --- |
| ⓪ structure | Structured operators | the Kronecker family, block, low-rank, Toeplitz and circulant, masked and interpolated operators | `_operators` |
| ⓪ structure | Tags and predicates | PSD, symmetric, Kronecker and the other tags, and what each unlocks | `_tags` |
| ⓪ structure | Sparse operators | `SparseOperator`, patterns, symbolic and numeric sparse Cholesky | `_sparse`, `_operators` |
| ⓪ structure | GMRF precisions | SPDE, random-walk, AR1, Besag and BYM2 builders | `_gmrf` |
| ⓪ structure | `add-operator` | extend | |
| ① primitive | Solves and factorisations | `solve`, `logdet`, `cholesky`, `sqrt`, `inv`, `diag`, `trace`, `eig`, `svd` | `_primitives` |
| ① primitive | Linear-algebra identities | Woodbury, Schur complements, Lyapunov, tridiagonal, selected inverse | `_linalg` |
| ① primitive | `add-dispatch-path` | extend | |
| ② strategy | Choosing a solver | dense, CG, preconditioned CG, BBMM, MINRES, LSMR, sparse Cholesky | `_strategies` |
| ② strategy | Log-determinant strategies | dense, stochastic Lanczos, Nyström | `_strategies` |
| ② strategy | Preconditioners | Jacobi, Nyström, partial Cholesky | `_preconditioners` |
| ② strategy | Integrators | Gauss–Hermite, unscented, cubature, Taylor, Monte Carlo | `_quadrature` |
| ② strategy | Passing a strategy through | how a library above gaussx exposes `solver=`; `linear_solve` for a bare matvec | `_strategies`, `_solve_frontend` |
| ② strategy | `add-solver-strategy` | extend | |
| setup | Randomized factorisations | randomized SVD and eigh, Nyström, RP-Cholesky, sketches | `_randomized`, `_sketching` |
| ③ model | Gaussians | densities, KL, entropy, conditioning, sampling | `_distributions` |
| ③ model | Distribution objects | `MultivariateNormal`, `LGSSM`, `MarkovGaussian`, with the numpyro extra | `_distributions` |
| ③ model | Exponential-family forms | natural, expectation and mean-covariance conversions | `_expfam` |
| ③ model | Likelihoods | Gaussian, Poisson, Bernoulli, Student-t and the rest | `_quadrature` |
| ③ model | SDE kernels | Matérn, periodic, quasi-periodic, sums and products | `_ssm` |
| ④ algorithm | GP conditioning and prediction | conditionals, prediction caches, LOVE, Kronecker GPs, leave-one-out | `_gp` |
| ④ algorithm | Variational GPs | ELBOs, the collapsed bound, whitening | `_gp` |
| ④ algorithm | Kalman filtering and smoothing | sequential, parallel and infinite-horizon | `_ssm` |
| ④ algorithm | Nonlinear filtering and sites | nonlinear Kalman, CVI site updates, SpInGP | `_ssm` |
| ④ algorithm | Expectations and moment matching | expected log-likelihood, EP moments, uncertain inputs | `_quadrature` |
| ④ algorithm | Natural-gradient and Newton updates | BLR, damped natural updates, Gauss–Newton | `_inference` |
| ④ algorithm | Ensemble Kalman | analysis, ETKF, localisation, inflation | `_inference` |
| ④ algorithm | `add-recipe` | extend | |
:::

gaussx has nothing at ⑤ or ⑥, so its shelf stops at algorithms.
Two entries are judgement calls: integrators sit at ② although they ship in `_quadrature` beside the likelihoods, and SDE kernels sit at ③ although they ship in `_ssm` with the Kalman code.
Both follow the role, not the subpackage, as jejjohnson/geostack#122 proposes for the integrators.

### Seven units: the shelf grouped into cards

A unit is a group of shelf entries that the same dependents use together. It is what a card describes and what a dependent's edge points at.
gaussx has seven, grouped by the band that works there.

[](#tbl-sg-units) lists them with their rungs, bands and subpackages.

Inside gaussx the units are read in ladder order: nouns, verbs, adverbs, then models and algorithms.
Two import tangles do not change that order.
Layers 0, 1 and 1.5 "form one dispatch core with import cycles", and `_distributions` uses the Kalman routines of `_ssm` (`AGENTS.md` @ `7fa1a38`).

### Three listed skills

Listing is a separate decision from the shelf: a listed skill is always in the agent's context, and a shelf entry is loaded when a listed skill or a card points at it.
A unit is listed when someone would ask for it without mentioning the rest.

:::{table} The listed use skills. Line budgets are proposals.
:label: tbl-sg-skills

| Skill | Units | Floor it carries cards for | Shelf entries it holds |
| --- | --- | --- | --- |
| `structured-gaussian-linalg`, the front door, at most 150 lines | structures and primitives | lineax, equinox | the four ⓪ entries and the two ① entries |
| `solvers-and-preconditioners`, at most 150 lines | strategies | structures and primitives; lineax | choosing a solver, log-determinant strategies, preconditioners, passing a strategy through, randomized factorisations |
| `gaussians-and-recipes`, at most 150 lines | Gaussians, and the four families | structures and primitives; strategies | the five ③ entries, the seven ④ entries and integrators |
:::

What each skill's body holds:

1. **`structured-gaussian-linalg`.** The three rules that are gaussx's own (operators not arrays, say what the structure is, never materialise to solve), the nouns and verbs a caller starts from, the "don't write it" table for ⓪ and ①, and an "enter at" table that routes to the other two skills.
2. **`solvers-and-preconditioners`.** How to choose between `AutoSolver`, `DenseSolver`, `CGSolver`, `PreconditionedCGSolver`, `BBMMSolver` and `SparseCholeskySolver`; which strategies take a `key=`; `linear_solve` for a bare `v ↦ A v`; and how a library above gaussx passes a strategy through, typed as `AbstractSolverStrategy`. pyrox-gp, kernellib and argminax all take strategies from gaussx.
3. **`gaussians-and-recipes`.** The Gaussian helpers (`gaussian_log_prob`, `gaussian_kl`, `conditional`, `sample_mvn`), the natural and expectation parameter conversions, that the distribution objects need the numpyro extra, and one "enter a family" table that points at the shelf entries.

### Seven cards

Each unit has one card, at most 15 lines, laid out by the grammar: nouns, verbs, adverbs.
The card for structures and primitives:

```markdown
- **Provides.** Covariance and precision operators that keep their structure, and the verbs that use it.
- **Nouns.** `gaussx.Kronecker`, `gaussx.BlockDiag`, `gaussx.BlockTriDiag`, `gaussx.Toeplitz`, `gaussx.SparseOperator`, `gaussx.low_rank_plus_diag`; any `lineax.AbstractLinearOperator`, tagged.
- **Verbs.** `gaussx.solve`, `gaussx.logdet`, `gaussx.cholesky`, `gaussx.sqrt`, `gaussx.inv`, `gaussx.diag`, `gaussx.trace`.
- **Adverbs.** `solver=`; left out, the verb dispatches on the structure.
- **Keep.** A dense matrix enters as `lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)`.
- **Never.** Call `as_matrix()` and then a dense `jnp.linalg` solve.
- **One rung down.** Open `gaussx:structured-gaussian-linalg` to build a new structure or to see which verb has a fast path for which noun.
```

A dependent package receives the card of each unit it uses.
The script reads that from the package's imports, with a threshold of three names (proposed, `CARD_THRESHOLD`).
Below the threshold the unit is named on an "also uses" line.

:::{table} The cards each dependent package would receive, and the units it uses fewer than three names of.
:label: tbl-sg-cards

<!-- shelf:cards:begin -->
| Dependent package | Cards | Also uses |
|---|---|---|
| pyrox-gp | Structures and primitives; Strategies; Gaussians; Gaussian processes; State space; Quadrature; Inference and ensembles |  |
| kernellib | Structures and primitives; Strategies |  |
| pyrox-lgm | Structures and primitives; Quadrature |  |
| argminax | Structures and primitives; Strategies | Inference and ensembles (2) |
| filterax | Structures and primitives; Inference and ensembles | Strategies (1); Gaussians (1); State space (1) |
| plumax | Structures and primitives | Strategies (1); Gaussians (2); State space (1) |
| gauss-flows | State space | Quadrature (2); Inference and ensembles (1) |
| radtrax | Structures and primitives | Strategies (1); Gaussians (1) |
| spectraldiffx | Structures and primitives |  |
| optax-bayes | Structures and primitives | Strategies (1) |
| manipy-jax | Structures and primitives |  |
| vardax | none | Structures and primitives (2); Inference and ensembles (1) |
| finitevolx | none | Strategies (2) |
<!-- shelf:cards:end -->
:::

The node is the package, not the repository.
pyrox is a workspace of four packages at this pin, and they differ: pyrox-lgm needs two cards, pyrox-nn and the core take no gaussx name, and pyrox-gp needs all seven.
pyrox-gp is the GP modelling shell and, at this pin, also holds the state-space models, which pyrox's default branch has since moved to a package of their own.
A workspace is then a small tree of its own: pyrox-gp builds on pyrox core, as gaussx's units build on each other.

### The extend skills

The four contributor skills already map onto the rungs; each gains two links and loses its restatement of what its role requires.

:::{table} gaussx's extend skills, which exist today in `.claude/skills/`.
:label: tbl-sg-extend

| Skill | Rung it adds to | Builds on (use unit) | Builds on (role file) |
| --- | --- | --- | --- |
| `add-operator` | ⓪ | structures and primitives | structure |
| `add-dispatch-path` | ① | structures and primitives | primitive |
| `add-solver-strategy` | ② | strategies | strategy |
| `add-recipe` | ③ ④ | Gaussians and the matching family | model, algorithm |
:::

They stay contributor skills, loaded inside the gaussx repository only.
`pre-pr-check`, `gaussx-review`, `triage-ci-failure`, `deprecate-or-rename` and the GitHub skills stay outside the tree.

### What gaussx needs from below

A third-party library gets a card in a use skill only when its types appear in that unit's public signatures.

:::{table} gaussx's third-party dependencies and where they surface. "Imported in" is from `src/gaussx` @ `7fa1a38`.
:label: tbl-sg-substrate

| Library | Imported in | Surfaces to a caller | Card in |
| --- | --- | --- | --- |
| lineax | every subpackage but `_sparse` | yes: every operator is an `AbstractLinearOperator`, and tags are lineax's | structures and primitives; strategies |
| equinox | throughout | yes: operators, strategies and models are Modules | covered by `role-ladder`'s kinds file |
| numpyro | `_distributions`, behind the extra | yes: `MultivariateNormal`, `LGSSM` and five more names exist only with it | Gaussians |
| matfree | `_primitives`, `_strategies`, `_linalg`, `_gp/_love.py` | no | `add-solver-strategy` only |
| optimistix | two primitives and the DARE solver | no | none |
:::

(sg-options)=
## Options

The options differ in which skills are listed, not in the shelf, the units or the cards.

### A. Three listed skills by band (recommended)

```text
plugins/gaussx/skills/                                                proposed (A)
  structured-gaussian-linalg/    SKILL.md  card.md  shelf/{operators,tags,sparse,gmrf,solves,identities}.md
  solvers-and-preconditioners/   SKILL.md  card.md  shelf/{solvers,logdet,preconditioners,pass-through,randomized}.md
  gaussians-and-recipes/         SKILL.md           shelf/{gaussians,distributions,expfam,likelihoods,sde-kernels,integrators}.md
                                                    shelf/{gp,variational-gp,kalman,nonlinear-filtering,expectations,natural-gradient,ensemble-kalman}.md
                                 cards/{gaussians,gp,state-space,quadrature,inference-and-ensembles}.md
```

### B. Seven listed skills, one per unit

```text
plugins/gaussx/skills/                                                proposed (B)
  structured-gaussian-linalg/  solvers-and-preconditioners/  gaussians/
  gaussian-processes/  state-space-and-kalman/  quadrature-and-expectations/  ensemble-and-natural-gradient-updates/
```

### C. One listed skill over the whole shelf

```text
plugins/gaussx/skills/                                                proposed (C)
  structured-gaussian-linalg/    SKILL.md  cards/*.md  references/*.md
```

:::{table} The options on the running example. The always-on cost uses the measured 170 tokens of today's listing.
:label: tbl-sg-options

| Criterion | A | B | C |
| --- | --- | --- | --- |
| Listed skills | 3 | 7 | 1 |
| Always in context | about 510 tokens (estimate) | about 1.2k tokens (estimate) | about 170 tokens, as today |
| kernellib's task: tune a preconditioner | loads `solvers-and-preconditioners` | the same | loads the front door, then a shelf entry |
| pyrox's task: a new likelihood | loads `gaussians-and-recipes`, then its likelihoods entry | loads `quadrature-and-expectations` | loads the front door, then a shelf entry |
| "Run a Kalman smoother" triggers | `gaussians-and-recipes`, by its description | its own skill | the front door, if its description says so |
| Cards | 7 | 7 | 7 |
:::

**Rejected.** One skill per rung: `_quadrature` spans three rungs, and the four families share rung ④ but not a caller.
**Rejected.** Splitting by subpackage: seventeen skills, and `_tags` or `_sparse` is never what someone asks for.

(sg-ownership)=
## Ownership

1. **gaussx** owns its shelf, its three listed skills and its seven cards. It never writes a "Builds on" block by hand.
2. **geostack** owns `role-ladder`, the substrate cards and the generator. It never edits gaussx's skill bodies.
3. **A dependent** owns nothing here: the cards it receives are decided by what it imports.

(sg-gates)=
## Gates

:::{table} The checks for gaussx's shelf. ST gates are those of the [skill tree](./skill_tree.md).
:label: tbl-sg-gates

| Gate | Passes when |
| --- | --- |
| GX-0 Current | `python scripts/skill_shelf.py --check` exits 0: the three generated tables match the submodules |
| GX-1 Covered | every name in `gaussx.__all__` is named by exactly one shelf entry, and every entry declares one role |
| GX-2 Cards name real symbols | every backticked `gaussx.<name>` in a card is in `gaussx.__all__` |
| GX-3 Budget | each `SKILL.md` at most 150 lines, each card at most 15 lines, each shelf entry at most 120 lines (proposed) |
| GX-4 Ladder | each unit declares its rungs, and no skill tells the agent to load a unit above it (ST-7, ST-4) |
| GX-5 Right part | on the two running-example tasks, the agent loads no gaussx skill for kernellib's first task and only `gaussians-and-recipes` for pyrox's (proposed) |
| GX-6 Nothing lost | every row of today's "Don't write it" table appears in exactly one of the three skills |
:::

(sg-risks)=
## Risks

- **The front door stops triggering for recipes.** Today one description covers Kalman filters and GPs. Fallback: `gaussians-and-recipes` takes those phrases into its own description, and GX-5 checks it.
- **The unit boundaries move.** jejjohnson/geostack#122 asks whether the Kalman scans move to filterax and whether `UnscentedIntegrator` is a strategy. Fallback: units follow subpackages, so a move changes one row of [](#tbl-sg-units) and regenerates the cards.
- **The import threshold hides a dependency that matters.** One name can be the whole reason for a dependency. Fallback: the "also uses" line names it, and a repository can pin a card in its shelf.
- **The counts undercount.** Only names in the static `__all__` of the package root are counted, so the seven names behind the numpyro extra and anything reached through a submodule are missed. Fallback: the gap is one direction only, so a card is never given for a unit a package does not use.
- **`role-ladder` drifts from its page.** Fallback: its vocabulary table is generated from the page's.

(sg-pattern)=
## The pattern for another repository

gaussx's shelf was derived in six steps, and each uses data the stack already has.

1. List the contributor skills. Each `add-<thing>` names something the library lets you add, so it names a rung the shelf must cover: gaussx's four are an operator, a dispatch path, a solver strategy and a recipe.
2. Count the public names by subpackage (`scripts/skill_shelf.py`), and give each subpackage its role.
3. Write the shelf: one entry per job a caller comes with, tagged with its role, following the role where it disagrees with the subpackage.
4. Count which names each dependent package takes (the same script), and group the entries into units where the dependents split.
5. List a unit when someone would ask for it alone.
6. Link each contributor skill to the entries it extends and to its role file, and give a third-party library a card only where its types surface.

A library whose dependents do not split keeps one unit and one listed skill, and still has a shelf.

(sg-followups)=
## Follow-up reports

- [ ] `feat(skills): move the role-ladder skill into geostack's plugin` — the root of every shelf; a first draft exists
- [ ] `feat(skills): write gaussx's shelf of 24 use skills and list three of them` — in gaussx; option A
- [ ] `docs(skills): seven contract cards for gaussx` — in gaussx; replaces the single gaussx card of jejjohnson/geostack#154
- [ ] `feat(scripts): choose a dependent's cards from the names it imports` — extends jejjohnson/geostack#152
- [ ] `docs(design): shelves for argminax and pyrox` — a small strategy library and a workspace, to test the pattern

## Relationships

- Parent: jejjohnson/geostack#150
- Related: jejjohnson/geostack#151 (the design this page works out for one library)
- Related: jejjohnson/geostack#122 (the role ladder, and the Kalman and integrator questions)
- Related: [skill tree](./skill_tree.md), [kinds and roles](./kinds_and_roles.md)
