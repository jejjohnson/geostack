# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Template.** Passages marked `<!-- template-example: … -->` in the agent files
  still describe the template's example; rewrite them from real code as it
  lands (the `adopt-template` skill).
- **No local package.** Reusable code goes upstream, not into `src/`: to the
  library that owns it, in the lowest layer that may hold it. The layers are
  GeoStack (pipekit, geotoolz, xrtoolz, georeader), GeoNumerics (gaussx,
  finitevolX, spectraldiffx), GeoML (geonnax, kernellib, optax_bayes,
  filterax, vardax, gauss_flows, xtremax, pyrox) and GeoModels (plumax,
  somax, radtrax), under `references/<layer>/`; edges between layers point
  down ("The layers" in `AGENTS.md`).
- **georeader is not ours.** It is spaceml-org's: read it, never write it,
  commit or push in it, or open issues and PRs on its repository.
  `.claude/settings.json` denies edits there and its PreToolUse hook
  (`.claude/hooks/guard_readonly.py`) refuses shell writes and `gh` writes;
  if one refuses, don't work round it, ask.
- **Reuse first.** Search the capability indexes (`docs/api/capabilities.md`
  for GeoStack, `capabilities_{numerics,geoml,geomodels}.md` for the others)
  before writing a loader, a model, a training step or a helper in a script
  or a notebook; the "Reuse before you write" table in `AGENTS.md` maps the
  usual hand-rolled code to what already exists.
- **Two environments.** `make` / `uv run` for tests, lint, types and docs;
  `pixi run` for experiments, DVC and anything CI runs. The layers above
  GeoStack have their own pixi environments (`pixi run -e geomodels …`).
- **No conda-forge here?** `pixi lock` needs it. Push the branch and let
  `pixi-lock.yml` commit `pixi.lock` and the indexes.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - adopting the template: `adopt-template`;
  - research workflow: `add-experiment`, `add-dataset`, `add-model`,
    `add-notebook`, `reproduce-results`, `write-up-results`;
  - shipping: `pre-pr-check`, `geostack-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `geostack-review`; run them on any diff that adds code, configs
  or pipeline stages, before committing:
  - `reuse-reviewer`: does the diff re-implement something in the package
    (`docs/api/capabilities.md`), a script, a config or a tool, or put
    reusable code in a script or notebook?
  - `reproducibility-reviewer`: seeds, hard-coded settings, incomplete DVC
    stages, leakage between splits, metrics written where nothing reads
    them, working-directory paths, results that cannot be regenerated.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
