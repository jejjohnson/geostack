---
title: "Giving a library a skill shelf"
subtitle: "Recipe · the files, the one shared check and the order of pull requests"
short_title: Skill shelf · recipe
description: "The steps that give a GeoStack library a shelf of use skills: the files of its plugin, the shelf file with its modules, skills, units and entries, the shelf check that geostack writes into the library, the library's own test, and the pull requests in order. gaussx is the worked example."
date: 2026-11-19
keywords:
  - agents
  - skills
  - skill tree
  - recipe
---

This page is the recipe behind [the skill tree](./skill_tree.md) for one library.
[gaussx's shelf](./skill_tree_gaussx.md) is the worked example, and every path below exists in gaussx.

## What a shelf is made of

All of it lives in the library, under `plugins/<name>/`.

```text
plugins/<name>/.claude-plugin/plugin.json      the plugin's name and description
plugins/<name>/tree/shelf.toml                 modules, listed skills, units, entries
plugins/<name>/tree/cards/<unit>.md            one contract card per unit, at most 15 lines
plugins/<name>/skills/<skill>/SKILL.md         a listed skill, at most 150 authored lines
plugins/<name>/skills/<skill>/shelf/<entry>.md one entry per job, at most 120 lines
scripts/skill_shelf.py                         the shelf check, written by geostack
tests/test_plugin_skill.py                     the library's test that runs it
```

An entry is one job a caller comes with, and it lists the public names it covers.
A unit is a group of entries that the same dependents use together, and it has one card.
A listed skill is what Claude Code shows; it holds the entries of one or more units.

## The shelf file

```toml
# Where the public names are read from: module = its file, from the root.
[modules]
gaussx = "src/gaussx/__init__.py"

[skills.structured-gaussian-linalg]
builds_on_units = []

[skills.solvers-and-preconditioners]
builds_on_units = ["structures-and-primitives"]

[units.structures-and-primitives]
title = "Structures and primitives"
rungs = "⓪ ①"
skill = "structured-gaussian-linalg"

[[entries]]
file = "operators"
title = "Structured operators"
role = "structure"
unit = "structures-and-primitives"
skill = "structured-gaussian-linalg"
covers = "the Kronecker family, block, low-rank and circulant operators"
names = ["Kronecker", "BlockDiag", "LowRankUpdate"]
```

- **`[modules]`.** One line per module whose `__all__` the shelf covers. With one module an entry lists bare names. With several, as in a workspace or a library of public submodules, it lists them with their module, as in `"functax.geometry.Affine"`. A submodule listed as a module of its own is covered by its names and is not a name of its parent.
- **`[skills.<skill>]`.** `builds_on_units` names the units of this library whose cards the skill carries. In a workspace, `packages = ["pyrox-gp"]` names the distributions the skill stands for, and its "Builds on" block is measured from them alone.
- **`role`.** One of structure, primitive, strategy, setup, model, algorithm, step and recipe, the rungs of the `role-ladder` skill.

## What is generated, and by whom

| Part of a `SKILL.md` | Markers | Written by |
|---|---|---|
| The table of the skill's entries | `<!-- shelf:begin -->` to `<!-- shelf:end -->` | the library, `python scripts/skill_shelf.py` |
| The cards of the units below the skill | `<!-- shelf:below:begin -->` to `<!-- shelf:below:end -->` | the library, the same command |
| The "Builds on" block | `<!-- skill-tree:begin … -->` to `<!-- skill-tree:end -->` | geostack, `python scripts/skill_tree.py --write <repository> --into <checkout>` |

`python scripts/skill_shelf.py --check` fails when a part is stale and on three gates.

| Gate | Holds when |
|---|---|
| SH-1 | every public name of every module is in exactly one entry, and the entry's file names it |
| SH-2 | every `module.X` a unit card writes right after a backtick is a public name |
| SH-3 | the authored part of a `SKILL.md` has at most 150 lines, an entry at most 120, a card at most 15 |

## The shelf check is one file

geostack holds it as `scripts/templates/skill_shelf.py`, and a library carries a copy as `scripts/skill_shelf.py`.
It reads the shelf file, the plugin's Markdown and the modules' source, and imports nothing, so it needs no configuration beyond `[modules]`.

1. Write the copy into a checkout of the library.

   ```bash
   python scripts/skill_tree.py --shelf-check <repository> --into <checkout>
   ```

2. Never edit the copy in the library. Change `scripts/templates/skill_shelf.py` here, then write it again into each library that carries it.

Gate ST-8 of `scripts/skill_tree.py --check` fails when a library's copy differs from geostack's at the pin.

## The library's test

`tests/test_plugin_skill.py` loads the script and asserts three things; gaussx's is the model.

- `violations(load(), public_names())` is empty, in the fast lane.
- Every `module.X` that any page of the plugin names is still public, in the fast lane.
- Every `python` block that `examples()` returns runs, in the slow lane.

## The order of pull requests

1. **In the library:** the shelf file, the entries, the cards, the listed skills, the copy of the shelf check and the test. Its own CI keeps it current from here on.
2. **In geostack:** the units in `SHELVES` of `scripts/skill_shelf.py`, so that dependents are measured against them, and the pin moved to the library's merged commit.
3. **In the library and in each dependent that has a skill:** the "Builds on" block, written with `--write`, after the pins of every library it reads are at their default branches.
4. **In geostack:** the pins moved to those commits, which turns gate ST-2 on for them.
