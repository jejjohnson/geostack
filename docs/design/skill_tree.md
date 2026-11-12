---
title: "A skill tree for agents: one skill per repository, linked along the dependency graph"
subtitle: "Design · agent skills that build on each other the way the libraries do"
short_title: Skill tree
description: "Proposes an agent skill tree for the GeoStack: one use skill per repository, linked to its parents along the generated dependency graph, with a short contract card per parent so an agent working high in the stack does not have to load the skills below it. Records how the geotoolz ⇄ pipekit cycle was broken and the three rules that keep the skill tree acyclic."
date: 2026-11-12
keywords:
  - agents
  - skills
  - plugins
  - dependency graph
  - skill tree
  - cycles
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

(st-decision)=
## Decision

Recommended: each repository ships **use skills by band of the role ladder** in its own plugin, a generated "Builds on" block links each to the units it depends on, one shared `role-ladder` skill sits at the root, and one geostack marketplace installs a skill together with its parents.
Alternative: one geostack plugin that holds a copy of every skill, which installs in one step but drifts from the repositories.
Status: open, 12 Nov 2026.
Revised the same day: the first draft had one skill per repository, which [](#st-tree) replaces with units by band; [gaussx's shelf](./skill_tree_gaussx.md) is the worked example.

What does not change: the dependency graph and its eight rules, the task skills in each repository's `.claude/skills/`, and the eight plugin skills that already exist.

(st-question)=
## Question

Can agent skills be layered like a video-game skill tree, so that a skill high in the stack references the skills below it instead of repeating them?
If so, what is a node, what is an edge, and what stops the tree from growing a cycle?

(st-motivation)=
## Motivation

The tree already exists as data; only the links between skills are missing.

- **The graph is generated.** `scripts/dependency_graph.py` @ `343d60e` holds `TIERS` (23 repositories in four layers) and reads 41 dependencies between repositories from each `pyproject.toml` and its imports. `tests/test_dependency_graph.py` fails on a cycle or a stale page.
- **Eight nodes are written.** Eight repositories ship a downstream plugin with one use skill and one reuse-reviewer agent ([](#tbl-st-nodes)). Fifteen ship none.
- **The existing skills do not know about each other.** kernellib's skill names gaussx 14 times in prose and pyrox's names it 5 times, but neither says which gaussx skill to load or what gaussx guarantees.
- **Each plugin is its own marketplace.** gaussx, kernellib, pyrox, geonnax and geotoolz each have a `.claude-plugin/marketplace.json` that lists one plugin, so installing pyrox's skill installs nothing below it.
- **An agent pays for what it loads.** The four skills on the path from pyrox to gaussx total 732 lines ([](#tbl-st-cost)). An agent fitting a GP rarely needs all of them.

:::{table} Repositories with a use skill today, at the pins of `343d60e`. Every repository except georeader also has an `AGENTS.md`.
:label: tbl-st-nodes

| Layer | Repository | Plugin skill | Pin |
| --- | --- | --- | --- |
| GeoNumerics | gaussx | `structured-gaussian-linalg` | `7fa1a38` |
| GeoNumerics | spectraldiffx | `spectral-methods-with-spectraldiffx` | `b3b61f6` |
| GeoNumerics | finitevolX | `finite-volume-models-with-finitevolx` | `32ffb45` |
| GeoML | geonnax | `build-models-with-geonnax` | `9579986` |
| GeoML | kernellib | `kernel-methods-with-kernellib` | `98fbbc7` |
| GeoML | xtremax | `extreme-value-analysis-with-xtremax` | `e3a5520` |
| GeoML | pyrox | `bayesian-models-with-pyrox` | `94eef13` |
| GeoStack | geotoolz | `build-geo-pipeline` | `9e92120` |
:::

No skill yet: functax, argminax, integrix, gauss_flows, optax_bayes, filterax, manipy, vardax, radtrax, somax, sesmax, plumax, pipekit, xrtoolz and georeader.

(st-example)=
## Running example: fitting a sparse GP with pyrox

An agent is asked to fit a sparse Gaussian process to 10 000 scattered observations with pyrox.
pyrox depends on kernellib, gaussx and geonnax as base dependencies and on gauss_flows as an extra.
All but gauss_flows have a skill today, so this path can be tested without writing a new node.

**TL;DR.** The agent loads the pyrox skill, reads three short cards about what its parents guarantee, and opens a parent skill only when the task reaches into it.

**Problem.** Let $V$ be the set of repositories and $E \subset V \times V$ the dependencies between them, where $(a, b) \in E$ reads "$a$ depends on $b$".
The graph $G = (V, E)$ is acyclic by rule 4 of the [dependency graph](../api/dependency_graph.md#dg-rules).
A skill tree assigns each repository $a$ a skill $s_a$ and a parent set $P(a) = \{b : (a, b) \in E\}$.
The cost of a task is the number of skill lines the agent loads, and the design goal is that a task inside $a$ loads $s_a$ and the cards of $P(a)$, not the skills of every ancestor.

:::{table} Lines an agent loads on the running example. Skill sizes are measured at the pins in [](#tbl-st-nodes); the card size is the proposed budget of [](#st-gates).
:label: tbl-st-cost

| What the agent loads | Lines |
| --- | --- |
| Today, all four skills on the path: pyrox 203, kernellib 181, geonnax 192, gaussx 156 | 732 |
| Proposed, pyrox's skill and three contract cards of at most 15 lines each | 248 (estimate) |
:::

```{mermaid}
flowchart TB
  subgraph geomodels["GeoModels"]
    direction LR
    plumax:::none
    radtrax:::none
    sesmax:::none
    somax:::none
  end
  subgraph geoml["GeoML"]
    direction LR
    pyrox:::has
    vardax:::none
    filterax:::none
    manipy:::none
    kernellib:::has
    optax_bayes:::none
    xtremax:::has
    gauss_flows:::none
    geonnax:::has
  end
  subgraph numerics["GeoNumerics"]
    direction LR
    finitevolX:::has
    argminax:::none
    integrix:::none
    spectraldiffx:::has
    functax:::none
    gaussx:::has
  end
  subgraph geostack["GeoStack"]
    direction LR
    xrtoolz:::none
    geotoolz:::has
    georeader:::none
    pipekit:::none
  end
  argminax --> gaussx
  filterax --> gaussx
  finitevolX --> spectraldiffx
  gauss_flows --> gaussx
  geotoolz --> georeader
  geotoolz --> pipekit
  integrix --> functax
  integrix --> gaussx
  kernellib --> gaussx
  kernellib --> geonnax
  manipy --> kernellib
  optax_bayes -.-> gaussx
  plumax --> finitevolX
  plumax --> pipekit
  pyrox -.-> gauss_flows
  pyrox --> kernellib
  radtrax --> gaussx
  radtrax -.-> georeader
  sesmax -.-> filterax
  sesmax --> finitevolX
  sesmax --> geonnax
  sesmax --> integrix
  sesmax --> pipekit
  sesmax -.-> vardax
  somax --> filterax
  somax --> finitevolX
  somax --> geonnax
  somax --> pipekit
  spectraldiffx --> gaussx
  vardax --> gaussx
  vardax --> geonnax
  vardax --> pipekit
  xrtoolz --> geotoolz
  classDef has stroke-width:3px
  classDef none stroke-dasharray:4 3
```

The diagram is the one on the [dependency graph](../api/dependency_graph.md#dg-graph) page, with a thick border on the eight repositories that have a skill today and a dashed border on the fifteen that do not.
It draws 33 of the 41 dependencies, because an arrow that a longer path implies is left out.
Each arrow reads "builds on", and the base skills sit at the bottom.

(st-tree)=
## The tree

A node is a unit: the rungs of one repository that one band of users works at. An edge is a dependency, and the root is a shared skill that explains the ladder once.

The [kinds-and-roles](./kinds_and_roles.md) page gives the principle: each kind of user "works in a band of the ladder and treats the rung below it as atomic: their floor".
A skill is written for a band, and a contract card is that band's floor.

1. **Shelf and units.** Each package has a shelf: one short skill per job a caller comes with, tagged with its role. [gaussx's shelf](./skill_tree_gaussx.md) has 24. A unit is a group of shelf entries that the same dependents use together; it is what a card describes and what an edge points at, and gaussx has seven. The node is the package, not the repository: pyrox is a workspace, and its packages use different parts of gaussx.
2. **Listing.** A unit is listed, and so always in the agent's context, when someone would ask for it without mentioning the rest. Every other shelf entry is loaded when a listed skill or a card points at it. Each listed skill costs about 170 tokens in every session (measured on the prototype).
3. **Edges.** Across repositories an edge is one of the 41 direct dependencies, and it points at the units the dependent uses, read from its imports. Inside a repository an edge follows the ladder: nouns, verbs, adverbs, then the rungs above.
4. **Contract cards.** One card per unit, at most 15 lines, laid out as nouns, verbs and adverbs. It says how to go exactly one rung down, and never further. The generator copies into a child's block only the cards of the units it uses.
5. **The root.** One `role-ladder` skill holds the ladder as grammar, one file per role, the three kinds and the bands. It is a listed skill of its own, because it also explains an algorithm or a library by the roles its pieces play. Every plugin depends on it, and no library's skill repeats it.
6. **Substrate.** A third-party library the stack stands on (lineax, equinox, diffrax, optimistix, numpyro, optax) gets a card where its types appear in a unit's public signatures. The cards live in geostack, because no single library owns them.
7. **Extend skills.** A repository's contributor skills (`add-operator`, `add-solver-strategy` and `add-recipe` in gaussx) build on the unit they extend and on the role file of the rung they add to. Maintenance skills stay outside the tree.
8. **Band front doors and capstones.** Geostack's plugin holds one entry-point skill per band, and a capstone per job that spans repositories, in the form of the running-example table of the kinds-and-roles page.

One skill per rung is still not the unit.
A skill's description is always in the agent's context, the rungs of one library can be cyclic (gaussx's layers 0, 1 and 1.5 "form one dispatch core with import cycles", `AGENTS.md` @ `7fa1a38`), and one subpackage can span three rungs.

(st-options)=
## Options

The options differ in what gets installed when an agent asks for one skill.
Every option needs the "Builds on" block, because `SKILL.md` has no frontmatter field for requiring another skill.

### A. Prose links only

Each skill gains a generated block, and nothing else changes.

```markdown
<!-- pyrox: plugins/pyrox/skills/bayesian-models-with-pyrox/SKILL.md -->
<!-- builds-on:begin · generated by geostack scripts/dependency_graph.py · proposed (A) -->
## Builds on

Load a parent skill only when the task changes how that library is used.
For calling it, the card below is enough.

### gaussx · `gaussx:structured-gaussian-linalg` · base dependency
Structured covariance and precision operators, and solves that keep the structure.
Start from `gaussx.solve`, `gaussx.logdet`, `gaussx.cholesky`, `Kronecker`, `LowRankUpdate`, `MultivariateNormal`.
Never densify an operator to call `jnp.linalg` on it.

### kernellib · `kernellib:kernel-methods-with-kernellib` · base dependency
<kernellib's card>

### geonnax · `geonnax:build-models-with-geonnax` · base dependency
<geonnax's card>

### gauss_flows · no skill yet · extra `[flows]`
<gauss_flows's card, from its AGENTS.md until it has a skill>
<!-- builds-on:end -->
```

The agent can follow the links only if the user installed the parents' plugins by hand, from four separate marketplaces.

### B. Plugin dependencies in one geostack marketplace (recommended)

The block of option A, and each plugin declares its parents the way its `pyproject.toml` does.

```json
// pyrox: plugins/pyrox/.claude-plugin/plugin.json                proposed (B)
{
  "name": "pyrox",
  "dependencies": ["gaussx", "kernellib", "geonnax"]
}
```

```json
// geostack: .claude-plugin/marketplace.json                      proposed (B)
{
  "name": "geostack",
  "plugins": [
    {"name": "gaussx", "source": {"source": "git-subdir", "url": "jejjohnson/gaussx", "path": "plugins/gaussx"}},
    {"name": "pyrox", "source": {"source": "git-subdir", "url": "jejjohnson/pyrox", "path": "plugins/pyrox"}}
  ]
}
```

Installing `pyrox@geostack` installs gaussx, kernellib and geonnax with it.
Only the four descriptions enter the agent's context; a skill body loads when the agent opens it.
One marketplace is needed because a dependency on a plugin in another marketplace is blocked unless the root marketplace allows it.

### C. One geostack plugin with every skill

geostack copies each repository's skill into `plugins/geostack/skills/` and ships them as one plugin.

```json
// geostack: .claude-plugin/marketplace.json                      proposed (C)
{"name": "geostack", "plugins": [{"name": "geostack", "source": "./plugins/geostack"}]}
```

One install gives the whole tree, and all 23 descriptions are always listed.
The copies follow geostack's pins, so a skill lags its repository until the pin moves.

:::{table} The options on the running example.
:label: tbl-st-options

| Criterion | A | B | C |
| --- | --- | --- | --- |
| Installed when the user asks for pyrox | pyrox | pyrox and its 3 parents with a skill | all 23 |
| Parent skill is there when the agent needs it | only if installed by hand | yes | yes |
| Where a skill is edited | its repository | its repository | its repository, then copied |
| Skill version matches the installed library | yes | yes | only at geostack's pin |
| New files | none | one marketplace, a `dependencies` field per plugin | one plugin, a sync step |
| Works for a repository used outside the stack | yes | yes, its own marketplace stays | no |
:::

**Rejected.** One skill per rung per repository: too many descriptions, and the rungs are cyclic inside gaussx.
**Rejected.** Hand-written "Builds on" blocks: they drift, as the prose mentions in today's skills already have.
**Rejected.** A single stack-wide skill: it puts every library's detail in context for every task, which is the cost the tree exists to avoid.

(st-cycle)=
## Breaking the cycle

The one cycle between repositories, geotoolz ⇄ pipekit, is already broken on `main`; this section records how, and the three rules that keep the skill tree free of cycles.

An earlier draft of this proposal said the cycle was still open.
That came from a checkout 54 commits behind `main`, pinned to pipekit `ef3c17f`, and is corrected here.

### How geotoolz ⇄ pipekit was broken

The cycle was one dataset that needed both sides.
pipekit-train's `XarrayWindowDataset` imported `geopatcher.time` from geotoolz-patcher (`xarray_window.py:35` and `_xreader.py:380` @ `ef3c17f`), and geotoolz depends on pipekit.
The import was lazy and guarded, so nothing failed at install time, and it was also broken: geotoolz had renamed the module to `geopatcher.temporal`.

The fix is rule 6 of the dependency graph, "a reverse edge is fixed by moving code, never by a shim", in its "move the adapter up" form.

1. **Add the adapter to the higher repository.** geotoolz gained `geopatcher.integrations.train`, behind the `geotoolz-patcher[train]` extra (jejjohnson/geotoolz#486, `f89118e`, 16 Oct 2026).
2. **Remove it from the lower one, with no re-export.** pipekit dropped `XarrayWindowDataset` and `_xreader` (jejjohnson/pipekit#57, `96c8c6b`, 16 Oct 2026). On pipekit `9e86a56` no package source imports `geopatcher`, `geotoolz`, `geocatalog` or `xrcore`.
3. **Move both pins together and shrink `KNOWN`.** geostack #116 (`75824ec`, 30 Oct 2026) moved the pipekit and geotoolz pins and deleted the `cycle` and `package-cycle` entries. The check now fails if the cycle returns.

On `main` @ `343d60e` the violation table lists two rows, both somax's undeclared filterax import, and no cycle.

The same three steps apply to any future cycle.
When a lower repository needs a higher one, either the primitive moves down or the adapter moves up into an `integrations/` module behind an extra, and the old path raises an `ImportError` naming the new home.

### Keeping the skill tree acyclic

A skill tree can grow a cycle in three ways the import graph cannot, and each has a rule.

1. **Edges are generated, never written.** The "Builds on" block and the `dependencies` field come from `scripts/dependency_graph.py`. The graph is acyclic by test, so the skill tree is too.
2. **A cycle inside a library is one node.** Where a library's rungs import each other, as gaussx's dispatch core does, they share a skill. Rungs are ordered by role in the "enter at" table, not by import.
3. **Load instructions point down.** A skill may tell the agent to load a parent, never a child. A parent may list who uses it under a "Used by" line, as a pointer only. Today geonnax's skill names pyrox 3 times and kernellib twice, and kernellib's names pyrox once; each becomes a "Used by" entry or goes.

(st-ownership)=
## Ownership

1. **Each repository** owns its use skill and its contract card. It never writes its own "Builds on" block or its plugin's `dependencies` by hand.
2. **geostack** owns the graph, the generator, the marketplace and the capstone skills. It never edits a repository's skill body.
3. **georeader** is third party, so its card lives in geostack. Nothing of ours moves into it (rule 7).

(st-gates)=
## Gates

:::{table} The checks that prove the tree holds.
:label: tbl-st-gates

| Gate | Passes when |
| --- | --- |
| ST-1 Acyclic | the skill graph has 0 cycles, and its edges equal the direct dependencies between repositories that have a skill |
| ST-2 Current | regenerating every "Builds on" block and `dependencies` field gives an empty diff |
| ST-3 Budget | each `SKILL.md` is under 500 lines, each description under 1 536 characters, each contract card at most 15 lines |
| ST-4 Downward | 0 load instructions name a skill of an equal or higher tier |
| ST-5 Installs | on a clean machine, installing `pyrox@geostack` installs exactly pyrox, kernellib, geonnax and gaussx |
| ST-6 Cards suffice | on 10 pyrox tasks that call gaussx without changing it, the agent opens the gaussx skill in at most 2 (proposed threshold) |
:::

The 500-line and 1 536-character limits are Claude Code's documented recommendation and cap for skills.
The 15-line card and the ST-6 threshold are proposals to tune on the running example.

(st-risks)=
## Risks

- **The agent ignores a "Builds on" link.** Nothing enforces it; a skill body naming another skill is a pattern that works in practice and is not documented behaviour (unverified). Fallback: the card carries the minimum, and option B guarantees the parent is installed.
- **A card drifts from the library.** A card is prose about an API that changes. Fallback: a card names only symbols in the repository's capability index, and ST-2 fails when one disappears (proposed).
- **Cross-repository sources are pinned differently from the submodules.** The marketplace takes a `ref` or `sha` per source, whose syntax was not checked (unverified). Fallback: the marketplace tracks each repository's default branch, and skills describe the latest release.
- **The listing budget.** Skill descriptions share a budget of 1% of the context window. The eight existing descriptions run 577 to 883 characters for the four measured. Fallback: shorten descriptions before adding rung-level skills.
- **Fifteen nodes are missing.** A tree with gaps sends the agent to an `AGENTS.md` instead. Fallback: the block falls back to the card alone, as for gauss_flows in option A.

(st-followups)=
## Follow-up reports

- [ ] `feat(scripts): emit each skill's builds-on block and plugin dependencies from the dependency graph` — the generator and gates ST-1, ST-2 and ST-4
- [ ] `feat(plugins): a geostack marketplace that lists every repository's plugin` — option B and gate ST-5
- [ ] `docs(skills): a contract card for each of the eight repositories with a skill` — bottom-up, gaussx first
- [ ] `[EPIC] a use skill for each of the fifteen repositories without one` — bottom-up by tier: functax, pipekit and gauss_flows first
- [ ] `[Design] capstone skills for the case studies` — which jobs get a skill and what they are called

## Relationships

- Related: #122 (the role ladder that the branches inside a node follow)
- Related: [dependency graph](../api/dependency_graph.md) (the edges and the rules)
- Related: [Report 16](../reports/16_geo_data_stack.md#r16-ws-d) (workstream D, where the cycle's fix was designed)
