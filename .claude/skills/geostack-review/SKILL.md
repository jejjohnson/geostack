---
name: geostack-review
description: Review a change or pull request in this research project against its own rules — CODE_REVIEW.md, the code-lives-upstream / config / pipeline / results / notebook / docs contracts in AGENTS.md, reuse of the upstream libraries instead of code in scripts and notebooks, and reproducibility. Use when asked to review a diff, branch or PR in this repo.
---

# Review a change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   what it touches: configs, scripts, `dvc.yaml`, notebooks, docs,
   dependencies, skills. When the diff comes as a file, check that it
   applies to the tree (its context lines are in the files it names)
   before reviewing against the tree, and say so if it does not.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: loaders,
   models, training loops, seeding and metrics re-implemented in scripts or
   notebooks are the main way a research codebase forks itself.
3. **Reproducibility** — run the `reproducibility-reviewer` subagent in
   parallel with step 2: seeds, hard-coded settings, incomplete DVC stages,
   leakage between splits, metrics written where nothing reads them,
   paths, results that cannot be regenerated.
4. **Contracts** — for each area touched, the rules in `AGENTS.md`:
   geostack has no package of its own, so code added under `src/` or a
   reusable function added to a script or a notebook is a finding (it goes
   upstream); every setting in `configs/`; stages declare deps / params / outs; numbers in the docs
   come from pipeline outputs; notebooks are `.py`, import from the
   upstream libraries and save the figures the docs use.
5. **Checklist** — the rest of `CODE_REVIEW.md`, skipping what ruff, ty or
   the tests already enforce.
6. **Verify claims** — for anything you flag as a bug, run it: the test,
   `pixi run dvc status` / `dvc repro <stage>`, the config composed with
   the override, the notebook top to bottom.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary), and end with what
you did and did not do: whether the two subagents ran or you applied their
checklists by hand, and which claims you ran and which you only read. A
claim you did not run is worded as unverified, not as fact.

A small, correct change gets a short review: say what you checked and what
does not apply, and add no suggestion for the sake of having one.
