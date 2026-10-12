# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Template.** Passages marked `<!-- template-example: … -->` in the agent files
  describe the bundled example (`myproject`); after adopting the template,
  rewrite them from your own code (the `adopt-template` skill).
- **Reuse first.** Search `docs/api/capabilities.md` (every public name in
  the package) before writing a loader, a model, a training step or a
  helper in a script or a notebook; the "Reuse before you write" table in
  `AGENTS.md` maps the usual hand-rolled code to what already exists.
- **Two environments.** `make` / `uv run` for tests, lint, types and docs;
  `pixi run` for experiments, DVC and anything CI runs.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - adopting the template: `adopt-template`;
  - research workflow: `add-experiment`, `add-dataset`, `add-model`,
    `add-notebook`, `reproduce-results`, `write-up-results`;
  - shipping: `pre-pr-check`, `myproject-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `myproject-review`; run them on any diff that adds code, configs
  or pipeline stages, before committing:
  - `reuse-reviewer`: does the diff re-implement something in the package
    (`docs/api/capabilities.md`), a script, a config or a tool, or put
    reusable code in a script or notebook?
  - `reproducibility-reviewer`: seeds, hard-coded settings, incomplete DVC
    stages, leakage between splits, metrics written where nothing reads
    them, working-directory paths, results that cannot be regenerated.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
