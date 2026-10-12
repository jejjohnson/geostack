---
name: myproject-review
description: Review a change or pull request in this research project against its own rules — CODE_REVIEW.md, the package / config / pipeline / results / notebook / docs contracts in AGENTS.md, reuse of the package instead of code in scripts and notebooks, and reproducibility. Use when asked to review a diff, branch or PR in this repo.
---

# Review a change

<!-- template-example: rename this skill to <your-project>-review after adoption (see adopt-template). -->

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   what it touches: the package, configs, scripts, `dvc.yaml`, notebooks,
   docs, dependencies.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: loaders,
   models, training loops, seeding and metrics re-implemented in scripts or
   notebooks are the main way a research codebase forks itself.
3. **Reproducibility** — run the `reproducibility-reviewer` subagent in
   parallel with step 2: seeds, hard-coded settings, incomplete DVC stages,
   leakage between splits, metrics written where nothing reads them,
   paths, results that cannot be regenerated.
4. **Contracts** — for each area touched, the rules in `AGENTS.md`: the
   package takes plain arguments and has tests; every setting in
   `configs/`; stages declare deps / params / outs; numbers in the docs
   come from pipeline outputs; notebooks are `.py`, import from the
   package and save the figures the docs use.
5. **Checklist** — the rest of `CODE_REVIEW.md`, skipping what ruff, ty or
   the tests already enforce.
6. **Verify claims** — for anything you flag as a bug, run it: the test,
   `pixi run dvc status` / `dvc repro <stage>`, the config composed with
   the override, the notebook top to bottom.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
